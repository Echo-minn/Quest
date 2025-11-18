import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import warnings
import os


# ------------------------------
# Config
# ------------------------------

TARGET_MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DRAFT_MODEL_ID  = "meta-llama/Llama-3.2-1B"  

DATASET_ID     = "THUDM/LongBench"
DATASET_SUBSET = "narrativeqa"

PAGE_SIZE       = 32
TOP_K_PAGES     = 10
N_DECODE_STEPS  = 16
PROMPT_MAX_LEN  = 2048

OUTPUT_DIR      = "outputs/kv_pages_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)



# ------------------------------
# Utility
# ------------------------------

def save_page_to_file(tokenizer, input_ids, page_idx, page_size, prefix):
    """Save one page of tokens to a human-readable text file."""
    start = page_idx * page_size
    end = start + page_size
    tokens = input_ids[0, start:end]

    text = tokenizer.decode(tokens, skip_special_tokens=False)
    filename = f"{OUTPUT_DIR}/{prefix}_page_{page_idx}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved page {page_idx} → {filename}")


def compute_page_scores(token_importance, page_size):
    """Convert per-token importance → per-page max importance."""
    L = len(token_importance)
    pad_len = (page_size - (L % page_size)) % page_size
    padded = F.pad(token_importance, (0, pad_len), value=-torch.inf)
    pages = padded.unfold(0, page_size, page_size).max(dim=1).values
    return pages


def load_model_and_tokenizer(model_id: str):
    print(f"Loading model: {model_id}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        print(f"Loaded {model_id}")
        return model, tokenizer

    except Exception as e:
        print(f"Failed to load {model_id}. Error: {e}")
        return None, None


# ------------------------------
# Core Experiment
# ------------------------------
def get_topk_pages_over_steps(model, tokenizer, input_ids, page_size, K, N):
    device = input_ids.device
    decoder_input = input_ids.clone()
    context_len = input_ids.shape[1]

    pages_found = set()

    for step in range(1, N+1):
        outputs = model.generate(
            decoder_input,
            max_new_tokens=1,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=model.config.eos_token_id
        )

        attn_layers = outputs.attentions[0]  # list of layers
        num_layers = len(attn_layers)

        # -------- MEMORY SAFE: accumulate layer means --------
        sum_layers = None
        for layer_attn in attn_layers:
            # layer_attn shape: [B, H, Q=1, S]
            layer_last = layer_attn[..., -1, :]  # → [B, H, S]
            if sum_layers is None:
                sum_layers = layer_last.float()
            else:
                sum_layers += layer_last.float()

        # mean over layers → [B, H, S]
        mean_layers = sum_layers / num_layers

        # mean over batch + heads → [S]
        token_imp = mean_layers.mean((0,1)).cpu()

        # only original prompt (ignore generated tokens)
        token_imp = token_imp[:context_len]

        # page pooling
        page_scores = compute_page_scores(token_imp, page_size)
        topk = torch.topk(page_scores, K).indices.tolist()
        
        print(f"Step {step}: pages = {topk}")
        pages_found.update(topk)

        # Save pages
        for p_idx in topk:
            save_page_to_file(
                tokenizer,
                input_ids,
                p_idx,
                page_size,
                prefix=f"step{step}_page{p_idx}"
            )

        decoder_input = outputs.sequences.to(device)
        
        del outputs, attn_layers, sum_layers, mean_layers, token_imp, page_scores
        # Tell PyTorch to release the cached memory
        torch.cuda.empty_cache()
        # --- END OF ADDED LINES ---

    return pages_found


def calculate_overlap(draft_pages, target_pages):
    draft = set(draft_pages)
    target = set(target_pages)

    inter = len(draft & target)
    union = len(draft | target)

    recall = inter / len(target) if len(target) else 0.0
    jaccard = inter / union if union else 0.0

    return recall, jaccard



# ------------------------------
# Main
# ------------------------------

def main():
    warnings.warn("Make sure DRAFT_MODEL_ID is real.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    target_model, target_tok = load_model_and_tokenizer(TARGET_MODEL_ID)
    draft_model, draft_tok   = load_model_and_tokenizer(DRAFT_MODEL_ID)

    if target_model is None or draft_model is None:
        print("Model load failed.")
        return

    # Load dataset
    print(f"Loading dataset {DATASET_ID}/{DATASET_SUBSET}...")
    ds = load_dataset(DATASET_ID, DATASET_SUBSET, split="test")

    sample = ds[0]
    prompt = sample['context'] + "\n\nQ: " + sample['input'] + "\nA:"

    # Tokenize
    encoded = target_tok(prompt, return_tensors="pt",
                         truncation=True, max_length=PROMPT_MAX_LEN)
    input_ids = encoded.input_ids.to(device)

    context_len = input_ids.shape[1]
    num_pages = (context_len + PAGE_SIZE - 1) // PAGE_SIZE

    print(f"Prompt = {context_len} tokens → {num_pages} pages")

    # Compute page sets
    print("\n=== Target Model Page Predictions ===")
    target_pages = get_topk_pages_over_steps(
        target_model, target_tok, input_ids, PAGE_SIZE, TOP_K_PAGES, N_DECODE_STEPS
    )

    print("\n=== Draft Model Page Predictions ===")
    draft_pages = get_topk_pages_over_steps(
        draft_model, target_tok, input_ids, PAGE_SIZE, TOP_K_PAGES, N_DECODE_STEPS
    )

    # Overlap
    recall, jaccard = calculate_overlap(draft_pages, target_pages)

    print("\n===== Final Results =====")
    print(f"Target pages: {sorted(target_pages)}")
    print(f"Draft pages:  {sorted(draft_pages)}")
    print("------------------------------")
    print(f"Recall : {recall:.4f}")
    print(f"Jaccard: {jaccard:.4f}")
    print("------------------------------")
    print("Interpretation: Recall = fraction of target-needed pages that the draft predicted.")



if __name__ == "__main__":
    main()
