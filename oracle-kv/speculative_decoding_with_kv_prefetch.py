import argparse
import dataclasses
import time
import numpy as np
import os
import torch
from tqdm.auto import tqdm

from transformers import AutoTokenizer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Assuming oracle_kv is available via the path append or installed
try:
    from oracle_kv import LlamaForCausalLM
except ImportError:
    # Fallback if running from project root without install
    from quest.models.llama import LlamaForCausalLM

@dataclasses.dataclass
class ModelConfig:
  model_path: str
  dtype: str = dataclasses.field(default="float16")
  device: str = dataclasses.field(default="cuda:0")

TARGET_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DRAFT_MODEL_ID  = "meta-llama/Llama-3.2-1B"  

DATASET_ID     = "THUDM/LongBench"
DATASET_SUBSET = "narrativeqa"
PROMPT_PATH = "./sample.prompt"

PAGE_SIZE       = 32
TOP_K_PAGES     = 10
DECODE_LEN  = 256
PROMPT_MAX_LEN  = 2048
TOKEN_BUDGET    = 256
DRAFT_AHEAD_LEN  = 16

OUTPUT_DIR      = "outputs/kv_pages_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model_and_tokenizer(model_cfg: ModelConfig):
    """Load model and tokenizer from pretrained model path."""
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    torch.set_default_dtype(dtype)

    with device:
        model = LlamaForCausalLM.from_pretrained(
            model_cfg.model_path,
            device_map=device,
            torch_dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def prefetch_kv(draft_model, target_model, draft_indices_history):
    """
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

def calculate_overlap(draft_k_out_indices, target_k_out_indices):
    # draft_k_out_indices, target_k_out_indices are lists of tensors or flat tensors
    if isinstance(draft_k_out_indices, list):
        if not draft_k_out_indices:
             draft = set()
        else:
             draft = set(torch.cat([x.view(-1) for x in draft_k_out_indices]).tolist())
    else:
        draft = set(draft_k_out_indices.view(-1).tolist())
        
    if isinstance(target_k_out_indices, list):
        if not target_k_out_indices:
             target = set()
        else:
             target = set(torch.cat([x.view(-1) for x in target_k_out_indices]).tolist())
    else:
        target = set(target_k_out_indices.view(-1).tolist())

    inter = len(draft & target)
    union = len(draft | target)
    
    recall = inter / len(target) if len(target) else 0.0
    jaccard = inter / union if union else 0.0

    return recall, jaccard

def main():
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading models on {device}...")
    try:
        target_model, target_tokenizer = load_model_and_tokenizer(ModelConfig(model_path=TARGET_MODEL_ID))
        draft_model, draft_tokenizer = load_model_and_tokenizer(ModelConfig(model_path=DRAFT_MODEL_ID))
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # We define the number of tokens in selected pages as the "Token Budget"
    target_model.quest_init(
        page_size=PAGE_SIZE,
        max_seq_len=PROMPT_MAX_LEN + DECODE_LEN,
        token_budget=TOKEN_BUDGET,
        dtype=dtype,
        device=device
    )
    draft_model.quest_init(
        page_size=PAGE_SIZE,
        max_seq_len=PROMPT_MAX_LEN + DECODE_LEN,
        token_budget=TOKEN_BUDGET,
        dtype=dtype,
        device=device
    )

    if not os.path.exists(PROMPT_PATH):
        with open(PROMPT_PATH, "w") as f:
            f.write("This is a sample prompt for testing speculative decoding. " * 50)

    with open(PROMPT_PATH, "r") as f:
        prompt = f.read()

    prompt_tokenized = target_tokenizer(prompt, return_tensors="pt")
    input_ids = prompt_tokenized.input_ids.to(device)
    context_len = input_ids.shape[1]
    
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

    pbar = tqdm(total=DECODE_LEN)
    
    cur_pos = context_len
    curr_input_ids = next_token
    
    while cur_pos < context_len + DECODE_LEN:
        start_time = time.time()
        
        # 1. Draft Phase
        draft_indices = []
        draft_tokens = []
        
        temp_input = curr_input_ids
        
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
            
            temp_input = next_draft_token
            draft_past_key_values = draft_out.past_key_values

        # 2. Prefetch KV for Target
        prefetch_kv(draft_model, target_model, draft_indices)
        
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
            
            t_pred = torch.argmax(t_out.logits[:, -1, :], dim=-1, keepdim=True)
            
            # Check for match
            if t_pred.item() == d_token.item():
                verified_count += 1
                t_input = d_token
            else:
                # Mismatch
                curr_input_ids = t_pred
                break
        else:
            # All matched
            curr_input_ids = t_pred
        
        # Stats
        if target_indices_list:
            draft_slice = draft_indices[:len(target_indices_list)]
            rec, jac = calculate_overlap(draft_slice, target_indices_list)
            all_recalls.append(rec)
            all_jaccards.append(jac)
        
        # 4. Rollback Draft Model
        # We generated DRAFT_AHEAD_LEN tokens in Draft Model.
        # We verified 'verified_count' tokens.
        # If all matched (verified_count == DRAFT_AHEAD_LEN), we don't need to rollback everything, 
        # but Draft Model state is now at 'end of draft sequence'.
        # Wait, if all matched, Draft Model is CORRECT up to end.
        # But if we want to continue drafting, we need to start from the new 'curr_input_ids'.
        
        # Case A: Mismatch at index 'i'.
        # Accepted tokens: 0 to i-1. (Count = i)
        # Draft generated: 0 to DRAFT_AHEAD_LEN-1.
        # We need to rollback Draft Model by (DRAFT_AHEAD_LEN - verified_count).
        
        tokens_to_rollback = DRAFT_AHEAD_LEN - verified_count
        if tokens_to_rollback > 0:
            draft_model.model.iController.rollback(tokens_to_rollback)
            # Also need to reset past_key_values wrapper if it tracks length?
            # LlamaForCausalLM uses iController for actual KV.
            # But the variable 'draft_past_key_values' holds reference to some tensor state?
            # In Quest, past_key_values is just a dummy or index list.
            # The real state is in iController. So calling rollback on iController is enough.
        
        # Advance
        n_advance = verified_count + 1
        cur_pos += n_advance
        pbar.update(n_advance)
        
        torch.cuda.synchronize()
        total_time += (time.time() - start_time)

    pbar.close()
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {DECODE_LEN / total_time:.2f} tokens/s")
    if all_recalls:
        print(f"Avg Recall: {np.mean(all_recalls):.4f}")
        print(f"Avg Jaccard: {np.mean(all_jaccards):.4f}")

if __name__ == "__main__":
    main()
