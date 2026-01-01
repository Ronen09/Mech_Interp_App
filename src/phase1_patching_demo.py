import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

torch.set_grad_enabled(False)

MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Choose a simple "fact" task where corruption plausibly flips the answer
CLEAN_PROMPT = "Delhi is the capital of"
CORRUPT_PROMPT = "Tokyo is the capital of"
TARGET_TOKEN_STR = " India"


def head_out_hook_factory(cache, layer: int, head: int):
    """
    Returns a hook that replaces the output of a single attention head
    with the cached clean value at the last position only.
    Hook point: blocks.{layer}.attn.hook_z has shape [batch, pos, head, d_head]
    """
    hook_name = f"blocks.{layer}.attn.hook_z"
    clean_head_result = cache[hook_name][:, -1, head, :].clone()

    def hook_fn(value: torch.Tensor, hook: HookPoint):
        value = value.clone()
        value[:, -1, head, :] = clean_head_result
        return value

    return hook_name, hook_fn


def patch_head_and_get_prob(model, corrupt_toks, clean_cache, target_id, layer, head):
    """Patch a single head and return probability of target token."""
    hook_name, hook_fn = head_out_hook_factory(clean_cache, layer, head)
    patched_logits = model.run_with_hooks(
        corrupt_toks, fwd_hooks=[(hook_name, hook_fn)]
    )
    patched_probs = patched_logits[0, -1].softmax(dim=-1)
    return patched_probs[target_id].item()


def sweep_all_heads(model, clean_cache, corrupt_toks, target_id, n_layers, n_heads):
    """Sweep all layer/head combinations and return results matrix."""
    results = np.zeros((n_layers, n_heads))

    total = n_layers * n_heads
    with tqdm(total=total, desc="Patching heads") as pbar:
        for layer in range(n_layers):
            for head in range(n_heads):
                prob = patch_head_and_get_prob(
                    model, corrupt_toks, clean_cache, target_id, layer, head
                )
                results[layer, head] = prob
                pbar.update(1)

    return results


def print_results_table(results, corrupt_prob, clean_prob):
    """Print a text-based heatmap of results."""
    n_layers, n_heads = results.shape

    # Calculate recovery ratio: how much of the gap from corrupt to clean is recovered
    recovery = (results - corrupt_prob) / (clean_prob - corrupt_prob + 1e-10)

    print("\n=== Patching Results (P(' France') after patching each head) ===")
    print(f"Baseline corrupt: {corrupt_prob:.4f}, clean: {clean_prob:.4f}")
    print()

    # Header
    print("       ", end="")
    for h in range(n_heads):
        print(f"  H{h:02d} ", end="")
    print()

    # Rows
    for layer in range(n_layers):
        print(f"L{layer:02d}  ", end="")
        for head in range(n_heads):
            prob = results[layer, head]
            rec = recovery[layer, head]
            # Color coding via symbols
            if rec > 0.1:
                marker = "█"  # High recovery
            elif rec > 0.05:
                marker = "▓"
            elif rec > 0.01:
                marker = "▒"
            else:
                marker = "░"
            print(f" {prob:.3f}", end="")
        print()

    print("\n=== Top 10 Most Important Heads (by recovery) ===")
    # Flatten and sort
    flat_indices = np.argsort(recovery.flatten())[::-1]
    for i, idx in enumerate(flat_indices[:10]):
        layer = idx // n_heads
        head = idx % n_heads
        prob = results[layer, head]
        rec = recovery[layer, head]
        print(
            f"{i + 1:2d}. L{layer:02d}H{head:02d}: P={prob:.4f}, recovery={rec * 100:.1f}%"
        )


def save_heatmap(results, corrupt_prob, clean_prob, filename="patching_heatmap.png"):
    """Save a matplotlib heatmap of results."""
    try:
        import matplotlib.pyplot as plt

        recovery = (results - corrupt_prob) / (clean_prob - corrupt_prob + 1e-10)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Raw probabilities
        im1 = axes[0].imshow(results, cmap="viridis", aspect="auto")
        axes[0].set_xlabel("Head")
        axes[0].set_ylabel("Layer")
        axes[0].set_title(
            f"P(' France') after patching\n(corrupt={corrupt_prob:.4f}, clean={clean_prob:.4f})"
        )
        plt.colorbar(im1, ax=axes[0], label="Probability")

        # Recovery ratio
        im2 = axes[1].imshow(
            recovery * 100, cmap="RdYlGn", aspect="auto", vmin=-5, vmax=20
        )
        axes[1].set_xlabel("Head")
        axes[1].set_ylabel("Layer")
        axes[1].set_title("Recovery % (how much of clean-corrupt gap is recovered)")
        plt.colorbar(im2, ax=axes[1], label="Recovery %")

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"\nHeatmap saved to {filename}")
        plt.close()
    except ImportError:
        print("\nmatplotlib not installed, skipping heatmap save")


def main():
    print(f"Loading {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    target_id = model.to_single_token(TARGET_TOKEN_STR)
    print(f"Using device: {DEVICE}")
    print(f"Model: {n_layers} layers, {n_heads} heads per layer")
    print(f"Target token: {TARGET_TOKEN_STR!r} -> id {target_id}")
    print(f"Clean prompt: {CLEAN_PROMPT!r}")
    print(f"Corrupt prompt: {CORRUPT_PROMPT!r}")

    # Cache clean activations
    clean_toks = model.to_tokens(CLEAN_PROMPT)
    _, clean_cache = model.run_with_cache(clean_toks)

    # Get baseline probabilities
    corrupt_toks = model.to_tokens(CORRUPT_PROMPT)
    corrupt_logits = model(corrupt_toks)
    corrupt_prob = corrupt_logits[0, -1].softmax(dim=-1)[target_id].item()

    clean_logits = model(clean_toks)
    clean_prob = clean_logits[0, -1].softmax(dim=-1)[target_id].item()

    print(f"\nBaseline P(' France'):")
    print(f"  Clean:   {clean_prob:.4f}")
    print(f"  Corrupt: {corrupt_prob:.4f}")
    print()

    # Sweep all heads
    results = sweep_all_heads(
        model, clean_cache, corrupt_toks, target_id, n_layers, n_heads
    )

    # Print results
    print_results_table(results, corrupt_prob, clean_prob)

    # Save heatmap
    save_heatmap(results, corrupt_prob, clean_prob)


if __name__ == "__main__":
    main()
