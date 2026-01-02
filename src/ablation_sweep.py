import numpy as np
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

torch.set_grad_enabled(False)

MODEL_NAME = "gpt2-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ablation test on clean prompt
CLEAN_PROMPT = "Paris is the capital of"
TARGET_TOKEN_STR = " France"


def ablate_head_hook_factory(layer: int, head: int):
    """
    Returns a hook that zeros out a single attention head's output.
    Hook point: blocks.{layer}.attn.hook_z has shape [batch, pos, head, d_head]
    """
    hook_name = f"blocks.{layer}.attn.hook_z"

    def hook_fn(value: torch.Tensor, hook: HookPoint):
        value = value.clone()
        value[:, -1, head, :] = 0.0
        return value

    return hook_name, hook_fn


def ablate_head_and_get_metrics(
    model, clean_toks, target_id, layer, head, ablate_all_positions=False
):
    """Ablate a single head and return probability, rank, entropy, and margin."""
    hook_name = f"blocks.{layer}.attn.hook_z"

    def hook_fn(value: torch.Tensor, hook: HookPoint):
        value = value.clone()
        if ablate_all_positions:
            value[:, :, head, :] = 0.0  # Ablate at all positions
        else:
            value[:, -1, head, :] = 0.0  # Ablate only at last position
        return value

    ablated_logits = model.run_with_hooks(clean_toks, fwd_hooks=[(hook_name, hook_fn)])
    ablated_probs = ablated_logits[0, -1].softmax(dim=-1)

    prob = ablated_probs[target_id].item()
    rank = (ablated_probs > ablated_probs[target_id]).sum().item() + 1
    entropy = -(ablated_probs * torch.log(ablated_probs + 1e-12)).sum().item()

    # Compute logit margin (more stable than rank)
    target_logit = ablated_logits[0, -1, target_id]
    other_logits = torch.cat(
        [ablated_logits[0, -1, :target_id], ablated_logits[0, -1, target_id + 1 :]]
    )
    margin = (target_logit - other_logits.max()).item()

    return prob, rank, entropy, margin


def sweep_all_heads(
    model, clean_toks, target_id, n_layers, n_heads, ablate_all_positions=False
):
    """Sweep all layer/head combinations and return results matrices."""
    probs = np.zeros((n_layers, n_heads))
    ranks = np.zeros((n_layers, n_heads), dtype=int)
    entropies = np.zeros((n_layers, n_heads))
    margins = np.zeros((n_layers, n_heads))

    total = n_layers * n_heads
    with tqdm(total=total, desc="Ablating heads") as pbar:
        for layer in range(n_layers):
            for head in range(n_heads):
                prob, rank, entropy, margin = ablate_head_and_get_metrics(
                    model, clean_toks, target_id, layer, head, ablate_all_positions
                )
                probs[layer, head] = prob
                ranks[layer, head] = rank
                entropies[layer, head] = entropy
                margins[layer, head] = margin
                pbar.update(1)

    return probs, ranks, entropies, margins


def print_results_table(
    delta_probs,
    delta_ranks,
    delta_entropies,
    delta_margins,
    baseline_prob,
    baseline_rank,
    baseline_margin,
):
    """Print results table with delta metrics."""
    n_layers, n_heads = delta_probs.shape

    print("\n=== Ablation Results (Delta Metrics) ===")
    print(
        f"Baseline: P={baseline_prob:.4f}, rank=#{baseline_rank}, margin={baseline_margin:.3f}"
    )
    print()

    print("\n=== Top 10 Heads by Δprob (probability drop) ===")
    flat_indices = np.argsort(delta_probs.flatten())[::-1]
    for i, idx in enumerate(flat_indices[:10]):
        layer = idx // n_heads
        head = idx % n_heads
        dp = delta_probs[layer, head]
        dr = delta_ranks[layer, head]
        de = delta_entropies[layer, head]
        dm = delta_margins[layer, head]
        print(
            f"{i + 1:2d}. L{layer:02d}H{head:02d}: Δprob={dp:.4f}, Δrank={dr:+d}, Δentropy={de:+.3f}, Δmargin={dm:.3f}"
        )

    print("\n=== Top 10 Heads by Δmargin (logit margin drop - most stable) ===")
    flat_indices = np.argsort(delta_margins.flatten())[::-1]
    for i, idx in enumerate(flat_indices[:10]):
        layer = idx // n_heads
        head = idx % n_heads
        dp = delta_probs[layer, head]
        dr = delta_ranks[layer, head]
        de = delta_entropies[layer, head]
        dm = delta_margins[layer, head]
        print(
            f"{i + 1:2d}. L{layer:02d}H{head:02d}: Δprob={dp:.4f}, Δrank={dr:+d}, Δentropy={de:+.3f}, Δmargin={dm:.3f}"
        )

    print("\n=== Top 10 Heads by Δrank (rank degradation) ===")
    flat_indices = np.argsort(delta_ranks.flatten())[::-1]
    for i, idx in enumerate(flat_indices[:10]):
        layer = idx // n_heads
        head = idx % n_heads
        dp = delta_probs[layer, head]
        dr = delta_ranks[layer, head]
        de = delta_entropies[layer, head]
        dm = delta_margins[layer, head]
        print(
            f"{i + 1:2d}. L{layer:02d}H{head:02d}: Δprob={dp:.4f}, Δrank={dr:+d}, Δentropy={de:+.3f}, Δmargin={dm:.3f}"
        )

    print("\n=== Top 10 Heads by Δentropy (distribution flattening) ===")
    flat_indices = np.argsort(delta_entropies.flatten())[::-1]
    for i, idx in enumerate(flat_indices[:10]):
        layer = idx // n_heads
        head = idx % n_heads
        dp = delta_probs[layer, head]
        dr = delta_ranks[layer, head]
        de = delta_entropies[layer, head]
        dm = delta_margins[layer, head]
        print(
            f"{i + 1:2d}. L{layer:02d}H{head:02d}: Δprob={dp:.4f}, Δrank={dr:+d}, Δentropy={de:+.3f}, Δmargin={dm:.3f}"
        )


def save_heatmap(
    delta_probs,
    delta_ranks,
    delta_entropies,
    delta_margins,
    filename="ablation_heatmap.png",
):
    """Save a matplotlib heatmap of delta metrics."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        # Δprob heatmap
        im1 = axes[0].imshow(delta_probs, cmap="Reds", aspect="auto", vmin=0)
        axes[0].set_xlabel("Head")
        axes[0].set_ylabel("Layer")
        axes[0].set_title("Δprob (probability drop)")
        plt.colorbar(im1, ax=axes[0], label="Δprob")

        # Δmargin heatmap (most stable metric)
        im2 = axes[1].imshow(delta_margins, cmap="Reds", aspect="auto")
        axes[1].set_xlabel("Head")
        axes[1].set_ylabel("Layer")
        axes[1].set_title("Δmargin (logit margin drop - most stable)")
        plt.colorbar(im2, ax=axes[1], label="Δmargin")

        # Δrank heatmap
        im3 = axes[2].imshow(delta_ranks, cmap="Reds", aspect="auto", vmin=0)
        axes[2].set_xlabel("Head")
        axes[2].set_ylabel("Layer")
        axes[2].set_title("Δrank (rank degradation)")
        plt.colorbar(im3, ax=axes[2], label="Δrank")

        # Δentropy heatmap
        im4 = axes[3].imshow(delta_entropies, cmap="RdBu_r", aspect="auto")
        axes[3].set_xlabel("Head")
        axes[3].set_ylabel("Layer")
        axes[3].set_title("Δentropy (distribution flattening)")
        plt.colorbar(im4, ax=axes[3], label="Δentropy")

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

    # Get baseline metrics (no ablation)
    clean_toks = model.to_tokens(CLEAN_PROMPT)
    baseline_logits = model(clean_toks)
    baseline_probs = baseline_logits[0, -1].softmax(dim=-1)
    baseline_prob = baseline_probs[target_id].item()
    baseline_rank = (baseline_probs > baseline_probs[target_id]).sum().item() + 1
    baseline_entropy = (
        -(baseline_probs * torch.log(baseline_probs + 1e-12)).sum().item()
    )

    # Compute baseline margin
    target_logit = baseline_logits[0, -1, target_id]
    other_logits = torch.cat(
        [baseline_logits[0, -1, :target_id], baseline_logits[0, -1, target_id + 1 :]]
    )
    baseline_margin = (target_logit - other_logits.max()).item()

    print(f"\nBaseline (no ablation):")
    print(f"  P(' Paris'): {baseline_prob:.4f}")
    print(f"  Rank: #{baseline_rank}")
    print(f"  Entropy: {baseline_entropy:.3f} nats")
    print(f"  Margin: {baseline_margin:.3f} logits")
    print()

    # Sweep all heads
    ablated_probs, ablated_ranks, ablated_entropies, ablated_margins = sweep_all_heads(
        model, clean_toks, target_id, n_layers, n_heads
    )

    # Calculate deltas
    delta_probs = baseline_prob - ablated_probs  # Positive = degradation
    delta_ranks = ablated_ranks - baseline_rank  # Positive = worse rank
    delta_entropies = ablated_entropies - baseline_entropy  # Positive = flatter
    delta_margins = (
        baseline_margin - ablated_margins
    )  # Positive = margin decreased (worse)

    # Print results
    print_results_table(
        delta_probs,
        delta_ranks,
        delta_entropies,
        delta_margins,
        baseline_prob,
        baseline_rank,
        baseline_margin,
    )

    # Save heatmap
    save_heatmap(delta_probs, delta_ranks, delta_entropies, delta_margins)


if __name__ == "__main__":
    main()
