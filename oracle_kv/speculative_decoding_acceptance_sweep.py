import time
import dataclasses
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

# Ensure local repo root is on sys.path *before* any site-packages so that the
# local `quest` package (with compiled `_kernels`) is used instead of any
# pip-installed version.
repo_root = os.path.join(os.path.dirname(__file__), "..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from oracle_kv.llama import LlamaForCausalLM


@dataclasses.dataclass
class ModelConfig:
    model_path: str
    dtype: str = dataclasses.field(default="float16")
    device: str = dataclasses.field(default="cuda:0")


TARGET_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DRAFT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

PAGE_SIZE = 32
DECODE_LEN = 256
PROMPT_MAX_LEN = 4096
TOKEN_BUDGET = 1024

# Sweep values for speculative window length
DRAFT_AHEAD_LEN_LIST: List[int] = [1, 2, 3, 4, 5, 8]


def load_model_and_tokenizer(model_cfg: ModelConfig):
    """Load model and tokenizer from pretrained model path."""
    device = torch.device(model_cfg.device)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
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


def build_prompt(tokenizer, target_ctx_tokens: int) -> torch.Tensor:
    """Build a single synthetic long prompt (no external dataset needed)."""
    print("Building synthetic long prompt...")
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


def run_one_sweep(
    target_model,
    draft_model,
    input_ids_cpu: torch.Tensor,
    device: torch.device,
    draft_ahead_len: int,
) -> Tuple[float, float]:
    """
    Run one speculative decoding experiment for a given DRAFT_AHEAD_LEN
    (without any KV prefetch) and return (throughput, avg_accepted_tokens_per_window).
    """
    # Reset KV-cache state so each sweep starts from a clean sequence.
    if hasattr(target_model, "quest_clear"):
        target_model.quest_clear()
    if hasattr(draft_model, "quest_clear"):
        draft_model.quest_clear()

    # Fresh copy of input_ids on device
    input_ids = input_ids_cpu.to(device)
    context_len = input_ids.shape[1]

    # Prefill stage
    print(f"\n[window={draft_ahead_len}] Prefilling...")
    with torch.no_grad():
        target_out = target_model(input_ids, use_cache=True)
        target_past_key_values = target_out.past_key_values

        draft_out = draft_model(input_ids, use_cache=True)
        draft_past_key_values = draft_out.past_key_values

        next_token = torch.argmax(target_out.logits[:, -1, :], dim=-1, keepdim=True)

    # Decode stage with speculative decoding (no prefetch)
    print(f"[window={draft_ahead_len}] Decoding...")

    total_time = 0.0
    verified_counts = []

    pbar = tqdm(total=DECODE_LEN)

    cur_pos = context_len
    curr_input_ids = next_token

    while cur_pos < context_len + DECODE_LEN:
        start_time = time.time()

        # 1. Draft phase
        draft_tokens = []
        temp_input = curr_input_ids

        for _ in range(draft_ahead_len):
            with torch.no_grad():
                draft_out = draft_model(
                    temp_input,
                    use_cache=True,
                    past_key_values=draft_past_key_values,
                )
            next_draft_token = torch.argmax(
                draft_out.logits[:, -1, :], dim=-1, keepdim=True
            )
            draft_tokens.append(next_draft_token)

            temp_input = next_draft_token
            draft_past_key_values = draft_out.past_key_values

        # 2. Target phase (verification)
        verified_count = 0
        t_input = curr_input_ids

        for d_token in draft_tokens:
            with torch.no_grad():
                t_out = target_model(
                    t_input, use_cache=True, past_key_values=target_past_key_values
                )

            target_past_key_values = t_out.past_key_values

            t_pred = torch.argmax(
                t_out.logits[:, -1, :], dim=-1, keepdim=True
            )

            if t_pred.item() == d_token.item():
                verified_count += 1
                t_input = d_token
            else:
                curr_input_ids = t_pred
                break
        else:
            # All matched: run one more target step to generate the next token
            with torch.no_grad():
                t_out = target_model(
                    t_input,
                    use_cache=True,
                    past_key_values=target_past_key_values,
                )
            target_past_key_values = t_out.past_key_values
            next_target_token = torch.argmax(
                t_out.logits[:, -1, :], dim=-1, keepdim=True
            )
            curr_input_ids = next_target_token

        # 3. Rollback draft model
        tokens_to_rollback = draft_ahead_len - verified_count
        if tokens_to_rollback > 0:
            draft_model.model.iController.rollback(tokens_to_rollback)

        # 4. Advance
        n_advance = verified_count + 1
        cur_pos += n_advance
        pbar.update(n_advance)

        torch.cuda.synchronize()
        total_time += time.time() - start_time
        verified_counts.append(verified_count)

    pbar.close()

    throughput = DECODE_LEN / total_time
    avg_accepted = float(np.mean(verified_counts)) if verified_counts else 0.0

    print(f"[window={draft_ahead_len}] Total time: {total_time:.2f}s")
    print(f"[window={draft_ahead_len}] Throughput: {throughput:.2f} tokens/s")
    print(f"[window={draft_ahead_len}] Avg Accepted tokens: {avg_accepted:.2f}")

    return throughput, avg_accepted


def main():
    dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading models on {device}...")
    try:
        target_model, target_tokenizer = load_model_and_tokenizer(
            ModelConfig(model_path=TARGET_MODEL_ID, dtype="float16", device=str(device))
        )
        draft_model, draft_tokenizer = load_model_and_tokenizer(
            ModelConfig(model_path=DRAFT_MODEL_ID, dtype="float16", device=str(device))
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # Quest KV-cache init: choose max window size for all sweeps
    max_window = max(DRAFT_AHEAD_LEN_LIST)
    max_seq_len = PROMPT_MAX_LEN + DECODE_LEN + max_window

    target_model.quest_init(
        page_size=PAGE_SIZE,
        max_seq_len=max_seq_len,
        token_budget=TOKEN_BUDGET,
        dtype=dtype,
        device=device,
    )
    draft_model.quest_init(
        page_size=PAGE_SIZE,
        max_seq_len=max_seq_len,
        token_budget=TOKEN_BUDGET,
        dtype=dtype,
        device=device,
    )

    # Build prompt once on CPU
    input_ids_cpu = build_prompt(
        target_tokenizer, target_ctx_tokens=min(PROMPT_MAX_LEN, 8000)
    )
    context_len = input_ids_cpu.shape[1]
    if context_len > PROMPT_MAX_LEN:
        input_ids_cpu = input_ids_cpu[:, -PROMPT_MAX_LEN:]
        context_len = PROMPT_MAX_LEN
    print(f"Prompt length: {context_len}")

    results = []
    for window in DRAFT_AHEAD_LEN_LIST:
        thr, avg_acc = run_one_sweep(
            target_model,
            draft_model,
            input_ids_cpu,
            device,
            draft_ahead_len=window,
        )
        results.append((window, thr, avg_acc))

    print("\n===== Summary over DRAFT_AHEAD_LEN sweep =====")
    for window, thr, avg_acc in results:
        print(
            f"window={window:2d}  throughput={thr:7.2f} tok/s  "
            f"AvgAccepted={avg_acc:.2f}"
        )


if __name__ == "__main__":
    main()


