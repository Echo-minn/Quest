import dataclasses
import time
import numpy as np
import os
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM

@dataclasses.dataclass
class ModelConfig:
  model_path: str
  dtype: str = dataclasses.field(default="float16")
  device: str = dataclasses.field(default="cuda:0")

TARGET_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DRAFT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"  

DECODE_LEN      = 256
PROMPT_MAX_LEN  = 4096
# How many tokens the draft model runs ahead before verification.
# Smaller values reduce wasted draft compute when acceptance rate is low.
DRAFT_AHEAD_LEN = 3

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
    # (capped by PROMPT_MAX_LEN to stay within the KV budget).
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

    input_ids = build_prompt(target_tokenizer, target_ctx_tokens=min(PROMPT_MAX_LEN, 8000))
    context_len = input_ids.shape[1]

    # Truncate the prompt if it's too long
    if context_len > PROMPT_MAX_LEN:
        input_ids = input_ids[:, -PROMPT_MAX_LEN:]
        context_len = PROMPT_MAX_LEN
    
    input_ids = input_ids.to(device)
    print(f"Prompt length: {context_len}")
    print(f"Draft ahead length: {DRAFT_AHEAD_LEN}")

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
    verified_counts = []

    # `tqdm` here is being used to show a progress bar as DECODE_LEN tokens are generated.
    # It creates a progress bar with DECODE_LEN steps, which will be updated during decoding.
    pbar = tqdm(total=DECODE_LEN)
    
    cur_pos = context_len
    curr_input_ids = next_token
    
    while cur_pos < context_len + DECODE_LEN:
        start_time = time.time()
        
        # 1. Draft Phase
        torch.cuda.nvtx.range_push("draft_block")
        draft_tokens = []
        
        # Save history of KV cache states for efficient rollback
        # draft_past_key_values initially contains state BEFORE processing temp_input
        draft_past_key_values_list = [draft_past_key_values]
        
        temp_input = curr_input_ids
        
        for _ in range(DRAFT_AHEAD_LEN):
            with torch.no_grad():
                draft_out = draft_model(
                    temp_input,
                    use_cache=True,
                    past_key_values=draft_past_key_values,
                )
            next_draft_token = torch.argmax(draft_out.logits[:, -1, :], dim=-1, keepdim=True)
            draft_tokens.append(next_draft_token)
            temp_input = next_draft_token
            draft_past_key_values = draft_out.past_key_values
            draft_past_key_values_list.append(draft_past_key_values)
        torch.cuda.nvtx.range_pop()
        
        # 2. Target Phase (Verification)
        torch.cuda.nvtx.range_push("target_verify")
        verified_count = 0
        t_input = curr_input_ids
        
        # We iterate over the draft tokens to verify them
        for i, d_token in enumerate(draft_tokens):
            with torch.no_grad():
                t_out = target_model(t_input, use_cache=True, past_key_values=target_past_key_values)
            
            target_past_key_values = t_out.past_key_values
            
            # Standard speculative decoding verification:
            # accept draft token iff it matches the greedy argmax of the target.
            t_pred = torch.argmax(t_out.logits[:, -1, :], dim=-1, keepdim=True)
            
            if t_pred.item() == d_token.item():
                verified_count += 1
                t_input = d_token
            else:
                # Mismatch: use target's prediction and re-sync draft model
                curr_input_ids = t_pred
                
                # Re-sync draft model using cached states
                # verified_count=0 means we reject the first draft token (D1).
                # We need to feed the correct token (T1) into the cache that generated D1.
                # The cache that generated D1 is the one AFTER processing the input token (T0).
                # This corresponds to draft_past_key_values_list[1].
                
                draft_past_key_values = draft_past_key_values_list[verified_count + 1]
                
                # Run on the corrected token to update cache for next step
                with torch.no_grad():
                    draft_sync_out = draft_model(curr_input_ids, use_cache=True, past_key_values=draft_past_key_values)
                draft_past_key_values = draft_sync_out.past_key_values
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
            next_target_token = torch.argmax(t_out.logits[:, -1, :], dim=-1, keepdim=True)
            curr_input_ids = next_target_token
            
            # Update draft model past_key_values: all draft tokens were accepted,
            # so we can use the draft model's past_key_values from the last draft step
            # (draft_past_key_values already contains all accepted tokens)
        torch.cuda.nvtx.range_pop()
        
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
    print(f"Avg Accepted tokens: {np.mean(verified_counts):.2f}")

if __name__ == "__main__":
    main()

