import argparse
import dataclasses
import time
import numpy as np
import os
import torch
from tqdm.auto import tqdm
from datasets import load_dataset

# Ensure local repo root is on sys.path *before* any site-packages so that the
# local `quest` package (with compiled `_kernels`) is used instead of any
# pip-installed version.
from transformers import AutoTokenizer, BitsAndBytesConfig
import sys
repo_root = os.path.join(os.path.dirname(__file__), "..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# Assuming oracle_kv is available via the path insert or installed
from oracle_kv.llama import LlamaForCausalLM

@dataclasses.dataclass
class ModelConfig:
  model_path: str
  dtype: str = dataclasses.field(default="float16")
  device: str = dataclasses.field(default="cuda:0")

TARGET_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DRAFT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

DATASET_ID     = "Salesforce/wikitext"
DATASET_SUBSET = "wikitext-2-v1"
PROMPT_PATH = "./sample.prompt"

PAGE_SIZE       = 32
TOP_K_PAGES     = 10
DECODE_LEN      = 256
PROMPT_MAX_LEN  = 4096
TOKEN_BUDGET    = 1024

DRAFT_AHEAD_LEN = 4

CONTEXT_LENS = [512, 1024, 2048, 4096, 8192]

OUTPUT_DIR      = "outputs/kv_pages_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model_and_tokenizer(model_cfg: ModelConfig):
    """Load model and tokenizer from pretrained model path."""
    device = torch.device(model_cfg.device)
    # We avoid changing the global default dtype here because it can interfere
    # with internal ops (e.g., attention mask construction) that assume a
    # float32 default. Dtype is controlled explicitly via model loading args.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = LlamaForCausalLM.from_pretrained(
        model_cfg.model_path,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def build_prefetch_maps(draft_model, target_model):
    """
    Build GPU lookup tables for Physical(Draft) -> Logical -> Physical(Target).
    Returns (draft_phys_to_logical, target_logical_to_phys) tensors.
    """
    device = draft_model.device
    
    # 1. Draft: Physical -> Logical
    draft_controller = draft_model.model.iController
    draft_capacity = draft_controller.kv_cache.pool.capacity
    
    # draft_controller.kv_cache.indicies is list: index=Logical, value=Physical
    # We want: index=Physical, value=Logical
    draft_phys_to_logical = torch.full((draft_capacity,), -1, device=device, dtype=torch.long)
    
    current_phys_indices_list = draft_controller.kv_cache.indicies
    if not current_phys_indices_list:
        return None, None
        
    current_phys_indices = torch.tensor(current_phys_indices_list, device=device, dtype=torch.long)
    current_logical_indices = torch.arange(len(current_phys_indices), device=device, dtype=torch.long)
    
    # Invert mapping
    draft_phys_to_logical[current_phys_indices] = current_logical_indices

    # 2. Target: Logical -> Physical
    # target_controller.kv_cache.indicies is list: index=Logical, value=Physical
    # We want to use this directly to map Logical -> Physical
    target_controller = target_model.model.iController
    target_phys_indices_list = target_controller.kv_cache.indicies
    target_logical_to_phys = torch.tensor(target_phys_indices_list, device=device, dtype=torch.long)
    
    return draft_phys_to_logical, target_logical_to_phys

def prefetch_step_async(draft_indices, draft_phys_to_logical, target_logical_to_phys, pool_buf, stream):
    """
    Async version of prefetch for a single step.
    Launch on the provided stream.
    """
    if draft_phys_to_logical is None or target_logical_to_phys is None:
        return

    with torch.cuda.stream(stream):
        # Flatten indices
        flat_indices = draft_indices.view(-1)
        
        # Filter valid physical indices (must be < capacity)
        valid_mask = flat_indices < len(draft_phys_to_logical)
        valid_indices = flat_indices[valid_mask]
        
        # Map to Logical
        logical_indices = draft_phys_to_logical[valid_indices]
        
        # Filter valid logical indices (mapped != -1)
        valid_logical_mask = logical_indices != -1
        logical_indices = logical_indices[valid_logical_mask]
        
        # Filter logical indices within target range
        valid_target_mask = logical_indices < len(target_logical_to_phys)
        final_logical = logical_indices[valid_target_mask]
        
        if final_logical.numel() == 0:
            return
            
        # Map to Target Physical
        target_physical = target_logical_to_phys[final_logical]
        
        # Touch pages (simple sum)
        # Optimization: We don't need to sum the whole block, just touching one element per cache line might be enough,
        # but sum() is a robust way to ensure read.
        selected_pages = pool_buf[0].index_select(0, target_physical)
        _ = selected_pages.sum()


def prefetch_kv(draft_model, target_model, draft_indices_history):
    """
    Legacy prefetch function:
    Given topk indices of draft model, prefetch the KV cache(touch the pages) for target model.
    Implements Option 2: Merge indices by frequency and retrieve 80% of the most frequent ones.
    """
    if not draft_indices_history:
        return

    # 1. Aggregate indices from all draft steps
    # indices are Physical Block Indices from the Draft Model
    all_indices = torch.cat([x.view(-1) for x in draft_indices_history]).long()
    
    # 2. Map Draft Physical -> Logical Page Index
    draft_controller = draft_model.model.iController
    # draft_controller.kv_cache.indicies is a list mapping Logical -> Physical
    
    draft_capacity = draft_controller.kv_cache.pool.capacity
    
    # Create a map tensor: index=Physical, value=Logical
    phys_to_logical = torch.full((draft_capacity,), -1, device=draft_model.device, dtype=torch.long)
    current_phys_indices = torch.tensor(draft_controller.kv_cache.indicies, device=draft_model.device, dtype=torch.long)
    current_logical_indices = torch.arange(len(current_phys_indices), device=draft_model.device, dtype=torch.long)
    phys_to_logical[current_phys_indices] = current_logical_indices
    
    # Filter valid indices
    valid_mask = (all_indices < draft_capacity) & (phys_to_logical[all_indices] != -1)
    valid_indices = all_indices[valid_mask]
    logical_indices = phys_to_logical[valid_indices]
    
    # 3. Frequency Count
    unique_logical, counts = torch.unique(logical_indices, return_counts=True)
    
    # Select top 80% most frequent
    if len(unique_logical) == 0:
        return

    # Sort by count descending
    sorted_idx = torch.argsort(counts, descending=True)
    num_to_fetch = max(1, int(len(unique_logical) * 0.8))
    top_logical = unique_logical[sorted_idx[:num_to_fetch]]
    
    # 4. Map Logical -> Target Physical
    target_controller = target_model.model.iController
    target_phys_list = target_controller.kv_cache.indicies
    target_phys_tensor = torch.tensor(target_phys_list, device=target_model.device, dtype=torch.long)
    
    # Ensure logical indices are within target's range
    valid_target_mask = top_logical < len(target_phys_tensor)
    final_logical = top_logical[valid_target_mask]
    target_physical = target_phys_tensor[final_logical]
    
    # 5. Touch Target Pages
    # Access the KV pool buffer
    # buf shape: (num_layers, capacity, 2, block_len, num_heads, head_dim)
    pool_buf = target_controller.kv_cache.pool.buf
    
    # We touch layer 0. Reading triggers cache fill.
    selected_pages = pool_buf[0].index_select(0, target_physical)
    _ = selected_pages.sum()

def _flatten_indices(indices):
    """Helper: flatten list-of-tensors or a single tensor into a Python list."""
    if isinstance(indices, list):
        if not indices:
            return []
        return torch.cat([x.view(-1) for x in indices]).tolist()
    else:
        return indices.view(-1).tolist()


def calculate_overlap(draft_k_out_indices, target_k_out_indices, draft_model, target_model):
    """
    Compute overlap between draft / target page selections in **logical page space**.

    Quest 的 `topk_dindices_buffer` 存的是 KV pool 的“物理页 index”：
      kv_cache.indicies[logical_idx] == physical_idx
    不同模型的物理页编号不对齐，所以这里先映射回逻辑页编号再做集合重合度，
    才能和 `spec-kv-validate-demo.py` 的 page overlap 保持语义一致。
    """
    draft_ctrl = draft_model.model.iController
    target_ctrl = target_model.model.iController

    # 物理页 -> 逻辑页 映射
    draft_phys_to_logical = {phys: logical for logical, phys in enumerate(draft_ctrl.kv_cache.indicies)}
    target_phys_to_logical = {phys: logical for logical, phys in enumerate(target_ctrl.kv_cache.indicies)}

    # 展平 top-k 物理页 index
    draft_phys_list = _flatten_indices(draft_k_out_indices)
    target_phys_list = _flatten_indices(target_k_out_indices)

    # 过滤掉当前缓存中不存在的物理块，映射到逻辑页
    draft_logical = {
        draft_phys_to_logical[p]
        for p in draft_phys_list
        if p in draft_phys_to_logical
    }
    target_logical = {
        target_phys_to_logical[p]
        for p in target_phys_list
        if p in target_phys_to_logical
    }

    inter = len(draft_logical & target_logical)
    union = len(draft_logical | target_logical)

    recall = inter / len(target_logical) if len(target_logical) else 0.0
    jaccard = inter / union if union else 0.0

    return recall, jaccard

def build_prompt(tokenizer, target_ctx_tokens: int):
    """Build a single synthetic long prompt (no external dataset needed) """
    
    print("Building synthetic long prompt...")
    base_paragraph = (
        "You are reading a long technical document about large language models, "
        "speculative decoding, and KV-cache paging. The text continues with detailed "
        "descriptions of algorithms, experiments, and implementation notes. "
        "In each section, the author explains how attention heads focus on different "
        "parts of the context, why some pages are more important than others, and how "
        "draft and target models may disagree on token predictions.\n"
    )

    # We iteratively repeat the paragraph until reaching a target token length
    # (capped by PROMPT_MAX_LEN to stay within the Quest KV budget).
    prompt_chunks = []
    cur_len = 0
    while cur_len < target_ctx_tokens:
        prompt_chunks.append(base_paragraph)
        tmp_prompt = "\n".join(prompt_chunks)
        encoded = tokenizer(tmp_prompt, return_tensors="pt")
        cur_len = encoded.input_ids.shape[1]
        if cur_len >= target_ctx_tokens:
            break

    return encoded.input_ids


def run_one_context(context_len_tokens: int):
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading models on {device}...")
    try:
        target_model, target_tokenizer = load_model_and_tokenizer(ModelConfig(model_path=TARGET_MODEL_ID))
        draft_model, draft_tokenizer = load_model_and_tokenizer(ModelConfig(model_path=DRAFT_MODEL_ID))
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # We define the number of tokens in selected pages as the "Token Budget".
    # `max_seq_len` must be large enough to hold:
    #   prefill context (<= PROMPT_MAX_LEN)
    # + planned decode length (DECODE_LEN)
    # + at most `DRAFT_AHEAD_LEN` extra tokens from the last speculative window.
    max_seq_len = PROMPT_MAX_LEN + DECODE_LEN + DRAFT_AHEAD_LEN
    target_model.quest_init(
        page_size=PAGE_SIZE,
        max_seq_len=max_seq_len,
        token_budget=TOKEN_BUDGET,
        dtype=dtype,
        device=device
    )
    draft_model.quest_init(
        page_size=PAGE_SIZE,
        max_seq_len=max_seq_len,
        token_budget=TOKEN_BUDGET,
        dtype=dtype,
        device=device
        )

    input_ids = build_prompt(target_tokenizer, target_ctx_tokens=context_len_tokens)
    context_len = input_ids.shape[1]

    # The LongBench context can be extremely long (e.g., >30k tokens), while our
    # Quest KV-Cache is initialized for at most PROMPT_MAX_LEN + DECODE_LEN tokens.
    # If we feed more tokens than max_seq_len, the KvPool will run out of free
    # blocks and raise `KeyError: 'pop from an empty set'`. To avoid this, we
    # truncate the prompt to the last PROMPT_MAX_LEN tokens before moving to GPU.
    if context_len > PROMPT_MAX_LEN:
        input_ids = input_ids[:, -PROMPT_MAX_LEN:]
        context_len = PROMPT_MAX_LEN
    
    input_ids = input_ids.to(device)
    print(f"Prompt length: {context_len}")

    # prefill stage
    print("Prefilling...")
    with torch.no_grad():
        target_out = target_model(input_ids, use_cache=True)
        target_past_key_values = target_out.past_key_values
        
        draft_out = draft_model(input_ids, use_cache=True)
        draft_past_key_values = draft_out.past_key_values
        
        next_token = torch.argmax(target_out.logits[:, -1, :], dim=-1, keepdim=True)

    # decode stage with speculative decoding
    print("Decoding...")
    
    total_time = 0
    all_recalls = []
    all_jaccards = []
    verified_counts = []

    # `tqdm` here is being used to show a progress bar as DECODE_LEN tokens are generated.
    # It creates a progress bar with DECODE_LEN steps, which will be updated during decoding.
    pbar = tqdm(total=DECODE_LEN)
    
    cur_pos = context_len
    curr_input_ids = next_token
    
    # Initialize prefetch stream
    prefetch_stream = torch.cuda.Stream(device=device)
    target_pool_buf = target_model.model.iController.kv_cache.pool.buf
    
    while cur_pos < context_len + DECODE_LEN:
        start_time = time.time()
        
        # 1. Draft Phase
        draft_indices = []
        draft_tokens = []
        
        temp_input = curr_input_ids
        
        # Build/Update maps for this decoding window (assuming existing pages don't change mapping)
        draft_phys_to_logical, target_logical_to_phys = build_prefetch_maps(draft_model, target_model)

        for _ in range(DRAFT_AHEAD_LEN):
            with torch.no_grad():
                draft_out = draft_model(
                    temp_input, 
                    use_cache=True,
                    past_key_values=draft_past_key_values
                )
                
            next_draft_token = torch.argmax(draft_out.logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens.append(next_draft_token)
            
            if hasattr(draft_model.model.iController, 'topk_dindices_buffer'):
                indices = draft_model.model.iController.topk_dindices_buffer.clone()
                draft_indices.append(indices)
                
                # Launch Async Prefetch
                # We need to wait for 'indices' to be ready on default stream
                prefetch_stream.wait_stream(torch.cuda.current_stream())
                prefetch_step_async(indices, draft_phys_to_logical, target_logical_to_phys, target_pool_buf, prefetch_stream)
            
            temp_input = next_draft_token
            draft_past_key_values = draft_out.past_key_values

        # 2. Prefetch KV for Target (Legacy/Fallback or ensuring coverage)
        # Since we did async per-step, this might be redundant but safer for "Option 2" compliance (frequency)
        # If we want pure speed and trust async, we can comment this out.
        # prefetch_kv(draft_model, target_model, draft_indices)
        
        # Wait for all prefetches to complete before Target Model runs?
        # Actually target model runs on default stream. 
        # We want prefetch to finish before Target Model needs the data.
        # Target model starts verifying now.
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        
        # 3. Target Phase (Verification)
        target_indices_list = []
        verified_count = 0
        t_input = curr_input_ids
        
        # We iterate over the draft tokens to verify them
        for i, d_token in enumerate(draft_tokens):
            with torch.no_grad():
                t_out = target_model(t_input, use_cache=True, past_key_values=target_past_key_values)
            
            target_past_key_values = t_out.past_key_values
            
            if hasattr(target_model.model.iController, 'topk_dindices_buffer'):
                target_indices_list.append(target_model.model.iController.topk_dindices_buffer.clone())
            
            # Standard speculative decoding verification:
            # accept draft token iff it matches the greedy argmax of the target.
            t_pred = torch.argmax(t_out.logits[:, -1, :], dim=-1, keepdim=True)
            
            if t_pred.item() == d_token.item():
                verified_count += 1
                t_input = d_token
            else:
                # Mismatch
                curr_input_ids = t_pred
                break
        else:
            # All matched: run one more target step to generate the next token
            with torch.no_grad():
                t_out = target_model(
                    t_input,
                    use_cache=True,
                    past_key_values=target_past_key_values
                )
            target_past_key_values = t_out.past_key_values
            if hasattr(target_model.model.iController, 'topk_dindices_buffer'):
                target_indices_list.append(target_model.model.iController.topk_dindices_buffer.clone())
            next_target_token = torch.argmax(t_out.logits[:, -1, :], dim=-1, keepdim=True)
            curr_input_ids = next_target_token
        
        # Stats (page overlap in logical page space)
        if target_indices_list:
            draft_slice = draft_indices[:len(target_indices_list)]
            rec, jac = calculate_overlap(draft_slice, target_indices_list, draft_model, target_model)
            all_recalls.append(rec)
            all_jaccards.append(jac)
        
        # 4. Rollback Draft Model
        tokens_to_rollback = DRAFT_AHEAD_LEN - verified_count
        if tokens_to_rollback > 0:
            draft_model.model.iController.rollback(tokens_to_rollback)
        
        # Advance
        n_advance = verified_count + 1
        cur_pos += n_advance
        pbar.update(n_advance)
        
        torch.cuda.synchronize()
        total_time += (time.time() - start_time)
        verified_counts.append(verified_count)
    pbar.close()
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {DECODE_LEN / total_time:.2f} tokens/s")
    # Accepted tokens
    # print(f"Accepted tokens: {verified_counts}")
    print(f"Avg Accepted tokens: {np.mean(verified_counts):.2f}")
    # Recall and Jaccard
    # print(f"Recall: {all_recalls}")
    print(f"Avg Recall: {np.mean(all_recalls):.4f}")
    # print(f"Jaccard: {all_jaccards}")
    print(f"Avg Jaccard: {np.mean(all_jaccards):.4f}")


def main():
    results = []
    for ctx in CONTEXT_LENS:
        print(f"\n===== Running with context_len = {ctx} tokens =====")
        thr, rec, jac = run_one_context(ctx)
        results.append((ctx, thr, rec, jac))

    print("\nSummary:")
    for ctx, thr, rec, jac in results:
        print(f"ctx={ctx:4d}  throughput={thr:7.2f} tok/s  "
              f"AvgRecall={rec:.4f}  AvgJaccard={jac:.4f}")

if __name__ == "__main__":
    main()

