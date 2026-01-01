import torch
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLEAN_PROMPT = "Paris is the capital of"
CORRUPT_PROMPT = "Berlin is the capital of"
TARGET_TOKEN_STR = " France"

# Head to patch
LAYER = 8
HEAD = 11


def main():
    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)

    target_id = model.to_single_token(TARGET_TOKEN_STR)
    print(f"Using device: {DEVICE}")
    print(f"Target token: {TARGET_TOKEN_STR!r} -> id {target_id}")
    print(f"Patching: L{LAYER}H{HEAD}")
    print()

    # Tokenize
    clean_toks = model.to_tokens(CLEAN_PROMPT)
    corrupt_toks = model.to_tokens(CORRUPT_PROMPT)

    # Cache clean activations
    _, clean_cache = model.run_with_cache(clean_toks)

    # Get baseline probabilities
    clean_prob = model(clean_toks)[0, -1].softmax(-1)[target_id].item()
    corrupt_prob = model(corrupt_toks)[0, -1].softmax(-1)[target_id].item()

    # Patch the head
    hook_name = f"blocks.{LAYER}.attn.hook_z"
    clean_head = clean_cache[hook_name][:, -1, HEAD, :].clone()

    def hook_fn(value, hook):
        value = value.clone()
        value[:, -1, HEAD, :] = clean_head
        return value

    patched_logits = model.run_with_hooks(corrupt_toks, fwd_hooks=[(hook_name, hook_fn)])
    patched_prob = patched_logits[0, -1].softmax(-1)[target_id].item()

    # Calculate recovery
    recovery = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob) * 100

    print(f"Clean P({TARGET_TOKEN_STR!r}):   {clean_prob:.4f}")
    print(f"Corrupt P({TARGET_TOKEN_STR!r}): {corrupt_prob:.4f}")
    print(f"Patched P({TARGET_TOKEN_STR!r}): {patched_prob:.4f}")
    print()
    print(f"L{LAYER}H{HEAD} Recovery: {recovery:.1f}%")


if __name__ == "__main__":
    main()
