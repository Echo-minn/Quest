import time
import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
import sys
import os
repo_root = os.path.join(os.path.dirname(__file__), "..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from oracle_kv.llama import LlamaForCausalLM

TARGET_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
PAGE_SIZE       = 32
DECODE_LEN      = 256
PROMPT_MAX_LEN  = 4096
TOKEN_BUDGET    = 1024

def load_model_and_tokenizer(model_id: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def build_prompt(tokenizer, target_ctx_tokens: int):
    base_paragraph = (
        "You are reading a long technical document about large language models, "
        "speculative decoding, and KV-cache paging. The text continues with detailed "
        "descriptions of algorithms, experiments, and implementation notes. "
        "In each section, the author explains how attention heads focus on different "
        "parts of the context, why some pages are more important than others, and how "
        "draft and target models may disagree on token predictions.\n"
    )
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model, target_tokenizer = load_model_and_tokenizer(TARGET_MODEL_ID)

    max_seq_len = PROMPT_MAX_LEN + DECODE_LEN
    target_model.quest_init(
        page_size=PAGE_SIZE,
        max_seq_len=max_seq_len,
        token_budget=TOKEN_BUDGET,
        dtype=torch.float16,
        device=device,
    )

    input_ids = build_prompt(target_tokenizer, target_ctx_tokens=min(PROMPT_MAX_LEN, 8000))
    context_len = input_ids.shape[1]
    if context_len > PROMPT_MAX_LEN:
        input_ids = input_ids[:, -PROMPT_MAX_LEN:]
        context_len = PROMPT_MAX_LEN
    input_ids = input_ids.to(device)
    print(f"Prompt length: {context_len}")

    print("Prefilling...")
    # prefill
    with torch.no_grad():
        out = target_model(input_ids, use_cache=True)
        past_key_values = out.past_key_values
        curr_input_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)


    # decode: target-only
    print("Decoding...")
    total_time = 0.0
    for _ in range(DECODE_LEN):

        start = time.time()
        with torch.no_grad():
            out = target_model(
                curr_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
        past_key_values = out.past_key_values
        curr_input_ids = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()
        total_time += time.time() - start

    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {DECODE_LEN / total_time:.2f} tokens/s")

if __name__ == "__main__":
    main()