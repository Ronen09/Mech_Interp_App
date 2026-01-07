import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformer_lens import HookedTransformer
from sae_lens import SAE

torch.set_grad_enabled(False)

app = FastAPI(title="GPT-2 Activation Patching")
templates = Jinja2Templates(directory="templates")

# Load model at startup
print("Loading gpt2-small...")
model = HookedTransformer.from_pretrained("gpt2-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

N_LAYERS = model.cfg.n_layers
N_HEADS = model.cfg.n_heads

# Load SAE for residual stream (layer 8)
print("Loading SAE for layer 8 residual stream...")
SAE_HOOK_POINT = "blocks.8.hook_resid_pre"
sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id=SAE_HOOK_POINT,
    device=device,
)[0]  # from_pretrained returns (sae, cfg, sparsity)
SAE_N_FEATURES = sae.cfg.d_sae
print(f"SAE loaded: {SAE_N_FEATURES} features at {SAE_HOOK_POINT}")


class TokenProb(BaseModel):
    token: str
    prob: float


class SweepRequest(BaseModel):
    clean_prompt: str = "Paris is the capital of"
    target_token: str = " France"
    clamp_token: str | None = None


class SweepResponse(BaseModel):
    delta_probs: list[list[float]]      # 12×12 array
    delta_ranks: list[list[float]]      # 12×12 array
    delta_entropies: list[list[float]]  # 12×12 array
    delta_margins: list[list[float]]    # 12×12 array
    clean_prob: float
    clean_rank: int
    top_clean: list[TokenProb]


class HeadSpec(BaseModel):
    layer: int
    head: int


class MultiAblateRequest(BaseModel):
    clean_prompt: str = "Paris is the capital of"
    target_token: str = " France"
    heads: list[HeadSpec] = []  # List of heads to ablate simultaneously


class MultiAblateResponse(BaseModel):
    clean_prompt: str
    target_token: str
    heads: list[HeadSpec]
    clean_prob: float
    ablated_prob: float
    clean_rank: int
    ablated_rank: int
    delta_prob: float
    delta_rank: int
    delta_entropy: float
    delta_margin: float
    top_clean: list[TokenProb]
    top_ablated: list[TokenProb]


class PatchSweepRequest(BaseModel):
    clean_prompt: str = "John gave Mary the book. Mary gave it to"
    corrupt_prompt: str = "Bob gave Mary the book. Bob gave it to"
    target_token: str = " John"


class PatchSweepResponse(BaseModel):
    clean_prompt: str
    corrupt_prompt: str
    target_token: str
    clean_prob: float
    corrupt_prob: float
    clean_rank: int
    corrupt_rank: int
    recovery_matrix: list[list[float]]  # 12x12 recovery percentages
    prob_matrix: list[list[float]]      # 12x12 patched probabilities
    top_clean: list[TokenProb]
    top_corrupt: list[TokenProb]


class MultiPatchRequest(BaseModel):
    clean_prompt: str = "John gave Mary the book. Mary gave it to"
    corrupt_prompt: str = "Bob gave Mary the book. Bob gave it to"
    target_token: str = " John"
    heads: list[HeadSpec] = []  # List of heads to patch simultaneously


class MultiPatchResponse(BaseModel):
    clean_prompt: str
    corrupt_prompt: str
    target_token: str
    heads: list[HeadSpec]
    clean_prob: float
    corrupt_prob: float
    patched_prob: float
    clean_rank: int
    corrupt_rank: int
    patched_rank: int
    recovery_pct: float
    top_clean: list[TokenProb]
    top_corrupt: list[TokenProb]
    top_patched: list[TokenProb]


class SAERequest(BaseModel):
    text: str = "The Eiffel Tower is located in Paris, France"
    top_k: int = 20  # Number of top features to return per token


class FeatureActivation(BaseModel):
    feature_index: int
    activation: float
    neuronpedia_url: str


class TokenFeatures(BaseModel):
    token: str
    position: int
    top_features: list[FeatureActivation]


class SAEResponse(BaseModel):
    text: str
    hook_point: str
    n_features: int
    tokens: list[TokenFeatures]
    top_features_overall: list[FeatureActivation]  # Top features across all tokens


class ResidualSweepRequest(BaseModel):
    clean_prompt: str = "John gave Mary the book. Mary gave it to"
    corrupt_prompt: str = "Bob gave Mary the book. Bob gave it to"
    target_token: str = " John"


class ResidualSweepResponse(BaseModel):
    clean_prompt: str
    corrupt_prompt: str
    target_token: str
    clean_prob: float
    corrupt_prob: float
    clean_rank: int
    corrupt_rank: int
    recovery_by_layer: list[float]  # Recovery percentage for each layer (0-11)
    prob_by_layer: list[float]      # Patched probability for each layer
    top_clean: list[TokenProb]
    top_corrupt: list[TokenProb]


class LayerSpec(BaseModel):
    layer: int


class MultiResidualPatchRequest(BaseModel):
    clean_prompt: str = "John gave Mary the book. Mary gave it to"
    corrupt_prompt: str = "Bob gave Mary the book. Bob gave it to"
    target_token: str = " John"
    layers: list[LayerSpec] = []  # List of layers to patch simultaneously


class MultiResidualPatchResponse(BaseModel):
    clean_prompt: str
    corrupt_prompt: str
    target_token: str
    layers: list[LayerSpec]
    clean_prob: float
    corrupt_prob: float
    patched_prob: float
    clean_rank: int
    corrupt_rank: int
    patched_rank: int
    recovery_pct: float
    top_clean: list[TokenProb]
    top_corrupt: list[TokenProb]
    top_patched: list[TokenProb]


class MLPSweepRequest(BaseModel):
    clean_prompt: str = "John gave Mary the book. Mary gave it to"
    corrupt_prompt: str = "Bob gave Mary the book. Bob gave it to"
    target_token: str = " John"


class MLPSweepResponse(BaseModel):
    clean_prompt: str
    corrupt_prompt: str
    target_token: str
    clean_prob: float
    corrupt_prob: float
    clean_rank: int
    corrupt_rank: int
    recovery_by_layer: list[float]  # Recovery percentage for each MLP layer (0-11)
    prob_by_layer: list[float]      # Patched probability for each MLP layer
    top_clean: list[TokenProb]
    top_corrupt: list[TokenProb]


class MultiMLPPatchRequest(BaseModel):
    clean_prompt: str = "John gave Mary the book. Mary gave it to"
    corrupt_prompt: str = "Bob gave Mary the book. Bob gave it to"
    target_token: str = " John"
    layers: list[LayerSpec] = []  # List of MLP layers to patch simultaneously


class MultiMLPPatchResponse(BaseModel):
    clean_prompt: str
    corrupt_prompt: str
    target_token: str
    layers: list[LayerSpec]
    clean_prob: float
    corrupt_prob: float
    patched_prob: float
    clean_rank: int
    corrupt_rank: int
    patched_rank: int
    recovery_pct: float
    top_clean: list[TokenProb]
    top_corrupt: list[TokenProb]
    top_patched: list[TokenProb]


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "n_layers": N_LAYERS, "n_heads": N_HEADS}


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS
    })


@app.post("/sweep", response_model=SweepResponse)
def sweep(request: SweepRequest):
    """Run ablation on all 144 heads (12 layers × 12 heads) and return delta metrics."""
    target_id = model.to_single_token(request.target_token)
    clean_toks = model.to_tokens(request.clean_prompt)

    # Get baseline (clean) metrics
    clean_logits = model(clean_toks)
    clean_probs = clean_logits[0, -1].softmax(-1)
    clean_prob = clean_probs[target_id].item()
    clean_rank = (clean_probs > clean_probs[target_id]).sum().item() + 1

    # Calculate baseline entropy and margin
    baseline_entropy = -(clean_probs * torch.log(clean_probs + 1e-12)).sum().item()
    target_logit = clean_logits[0, -1, target_id]
    other_logits = torch.cat([
        clean_logits[0, -1, :target_id],
        clean_logits[0, -1, target_id + 1:]
    ])
    baseline_margin = (target_logit - other_logits.max()).item()

    # Initialize result arrays
    delta_probs = []
    delta_ranks = []
    delta_entropies = []
    delta_margins = []

    # Sweep all 144 heads
    for layer in range(N_LAYERS):
        layer_delta_probs = []
        layer_delta_ranks = []
        layer_delta_entropies = []
        layer_delta_margins = []

        for head in range(N_HEADS):
            # Create ablation hook
            hook_name = f"blocks.{layer}.attn.hook_z"

            def hook_fn(value, hook):
                value = value.clone()
                value[:, -1, head, :] = 0.0
                return value

            # Run ablation
            ablated_logits = model.run_with_hooks(
                clean_toks, fwd_hooks=[(hook_name, hook_fn)]
            )

            # Optional: Clamp selected token logit to baseline
            if request.clamp_token:
                try:
                    clamp_id = model.to_single_token(request.clamp_token)
                    baseline_clamp_logit = clean_logits[0, -1, clamp_id].item()
                    ablated_logits[0, -1, clamp_id] = baseline_clamp_logit
                except:
                    pass

            # Calculate metrics
            ablated_probs = ablated_logits[0, -1].softmax(-1)
            ablated_prob = ablated_probs[target_id].item()
            ablated_rank = (ablated_probs > ablated_probs[target_id]).sum().item() + 1

            # Entropy
            ablated_entropy = -(ablated_probs * torch.log(ablated_probs + 1e-12)).sum().item()

            # Margin
            target_logit = ablated_logits[0, -1, target_id]
            other_logits = torch.cat([
                ablated_logits[0, -1, :target_id],
                ablated_logits[0, -1, target_id + 1:]
            ])
            ablated_margin = (target_logit - other_logits.max()).item()

            # Calculate deltas
            delta_prob = clean_prob - ablated_prob
            delta_rank = ablated_rank - clean_rank
            delta_entropy = ablated_entropy - baseline_entropy
            delta_margin = baseline_margin - ablated_margin

            layer_delta_probs.append(delta_prob)
            layer_delta_ranks.append(float(delta_rank))
            layer_delta_entropies.append(delta_entropy)
            layer_delta_margins.append(delta_margin)

        delta_probs.append(layer_delta_probs)
        delta_ranks.append(layer_delta_ranks)
        delta_entropies.append(layer_delta_entropies)
        delta_margins.append(layer_delta_margins)

    # Get top-10 clean tokens for clamping
    def get_top_k(probs, k=10):
        vals, idx = probs.topk(k)
        tokens = [model.to_string(i.item()) for i in idx]
        return [TokenProb(token=t, prob=p.item()) for t, p in zip(tokens, vals)]

    top_clean = get_top_k(clean_probs)

    return SweepResponse(
        delta_probs=delta_probs,
        delta_ranks=delta_ranks,
        delta_entropies=delta_entropies,
        delta_margins=delta_margins,
        clean_prob=clean_prob,
        clean_rank=clean_rank,
        top_clean=top_clean,
    )


@app.post("/multi-ablate", response_model=MultiAblateResponse)
def multi_ablate(request: MultiAblateRequest):
    """Ablate multiple heads simultaneously and measure combined impact."""
    target_id = model.to_single_token(request.target_token)
    clean_toks = model.to_tokens(request.clean_prompt)

    # Get baseline (clean) metrics
    clean_logits = model(clean_toks)
    clean_probs = clean_logits[0, -1].softmax(-1)
    clean_prob = clean_probs[target_id].item()
    clean_rank = (clean_probs > clean_probs[target_id]).sum().item() + 1

    # Calculate baseline entropy and margin
    baseline_entropy = -(clean_probs * torch.log(clean_probs + 1e-12)).sum().item()
    target_logit = clean_logits[0, -1, target_id]
    other_logits = torch.cat([
        clean_logits[0, -1, :target_id],
        clean_logits[0, -1, target_id + 1:]
    ])
    baseline_margin = (target_logit - other_logits.max()).item()

    # Create hooks for all heads to ablate
    hooks = []
    for head_spec in request.heads:
        hook_name = f"blocks.{head_spec.layer}.attn.hook_z"
        head_idx = head_spec.head

        def make_hook(h_idx):
            def hook_fn(value, hook):
                value = value.clone()
                value[:, -1, h_idx, :] = 0.0
                return value
            return hook_fn

        hooks.append((hook_name, make_hook(head_idx)))

    # Run with all ablations applied simultaneously
    if hooks:
        ablated_logits = model.run_with_hooks(clean_toks, fwd_hooks=hooks)
    else:
        ablated_logits = clean_logits

    ablated_probs = ablated_logits[0, -1].softmax(-1)
    ablated_prob = ablated_probs[target_id].item()
    ablated_rank = (ablated_probs > ablated_probs[target_id]).sum().item() + 1

    # Calculate ablated entropy and margin
    ablated_entropy = -(ablated_probs * torch.log(ablated_probs + 1e-12)).sum().item()
    ablated_target_logit = ablated_logits[0, -1, target_id]
    ablated_other_logits = torch.cat([
        ablated_logits[0, -1, :target_id],
        ablated_logits[0, -1, target_id + 1:]
    ])
    ablated_margin = (ablated_target_logit - ablated_other_logits.max()).item()

    # Calculate deltas
    delta_prob = clean_prob - ablated_prob
    delta_rank = ablated_rank - clean_rank
    delta_entropy = ablated_entropy - baseline_entropy
    delta_margin = baseline_margin - ablated_margin

    # Get top-10 tokens
    def get_top_k(probs, k=10):
        vals, idx = probs.topk(k)
        tokens = [model.to_string(i.item()) for i in idx]
        return [TokenProb(token=t, prob=p.item()) for t, p in zip(tokens, vals)]

    return MultiAblateResponse(
        clean_prompt=request.clean_prompt,
        target_token=request.target_token,
        heads=request.heads,
        clean_prob=clean_prob,
        ablated_prob=ablated_prob,
        clean_rank=clean_rank,
        ablated_rank=ablated_rank,
        delta_prob=delta_prob,
        delta_rank=delta_rank,
        delta_entropy=delta_entropy,
        delta_margin=delta_margin,
        top_clean=get_top_k(clean_probs),
        top_ablated=get_top_k(ablated_probs),
    )


@app.post("/multi-patch", response_model=MultiPatchResponse)
def multi_patch(request: MultiPatchRequest):
    """Patch multiple heads simultaneously with clean activations and measure combined recovery."""
    target_id = model.to_single_token(request.target_token)
    clean_toks = model.to_tokens(request.clean_prompt)
    corrupt_toks = model.to_tokens(request.corrupt_prompt)

    # Cache clean activations
    _, clean_cache = model.run_with_cache(clean_toks)

    # Get baseline probabilities
    clean_logits = model(clean_toks)
    clean_probs = clean_logits[0, -1].softmax(-1)
    clean_prob = clean_probs[target_id].item()
    clean_rank = (clean_probs > clean_probs[target_id]).sum().item() + 1

    corrupt_logits = model(corrupt_toks)
    corrupt_probs = corrupt_logits[0, -1].softmax(-1)
    corrupt_prob = corrupt_probs[target_id].item()
    corrupt_rank = (corrupt_probs > corrupt_probs[target_id]).sum().item() + 1

    # Create hooks for all heads to patch
    hooks = []
    for head_spec in request.heads:
        hook_name = f"blocks.{head_spec.layer}.attn.hook_z"
        clean_head_activation = clean_cache[hook_name][:, -1, head_spec.head, :].clone()

        def make_patch_hook(clean_act, h_idx):
            def hook_fn(value, hook):
                value = value.clone()
                value[:, -1, h_idx, :] = clean_act
                return value
            return hook_fn

        hooks.append((hook_name, make_patch_hook(clean_head_activation, head_spec.head)))

    # Run corrupt prompt with all patches applied simultaneously
    if hooks:
        patched_logits = model.run_with_hooks(corrupt_toks, fwd_hooks=hooks)
    else:
        patched_logits = corrupt_logits

    patched_probs = patched_logits[0, -1].softmax(-1)
    patched_prob = patched_probs[target_id].item()
    patched_rank = (patched_probs > patched_probs[target_id]).sum().item() + 1

    # Calculate recovery percentage
    if abs(clean_prob - corrupt_prob) > 1e-10:
        recovery_pct = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob) * 100
    else:
        recovery_pct = 0.0

    # Get top-10 tokens
    def get_top_k(probs, k=10):
        vals, idx = probs.topk(k)
        tokens = [model.to_string(i.item()) for i in idx]
        return [TokenProb(token=t, prob=p.item()) for t, p in zip(tokens, vals)]

    return MultiPatchResponse(
        clean_prompt=request.clean_prompt,
        corrupt_prompt=request.corrupt_prompt,
        target_token=request.target_token,
        heads=request.heads,
        clean_prob=clean_prob,
        corrupt_prob=corrupt_prob,
        patched_prob=patched_prob,
        clean_rank=clean_rank,
        corrupt_rank=corrupt_rank,
        patched_rank=patched_rank,
        recovery_pct=recovery_pct,
        top_clean=get_top_k(clean_probs),
        top_corrupt=get_top_k(corrupt_probs),
        top_patched=get_top_k(patched_probs),
    )


@app.post("/patch-sweep", response_model=PatchSweepResponse)
def patch_sweep(request: PatchSweepRequest):
    """
    Patch sweep: For each head, patch its clean activation into the corrupt run
    and measure recovery of target token probability.

    This is useful for variable binding analysis - identifying which heads
    store/pass information about entities (names, objects) across the context.
    """
    target_id = model.to_single_token(request.target_token)
    clean_toks = model.to_tokens(request.clean_prompt)
    corrupt_toks = model.to_tokens(request.corrupt_prompt)

    # Cache clean activations
    _, clean_cache = model.run_with_cache(clean_toks)

    # Get baseline probabilities
    clean_logits = model(clean_toks)
    clean_probs = clean_logits[0, -1].softmax(-1)
    clean_prob = clean_probs[target_id].item()
    clean_rank = (clean_probs > clean_probs[target_id]).sum().item() + 1

    corrupt_logits = model(corrupt_toks)
    corrupt_probs = corrupt_logits[0, -1].softmax(-1)
    corrupt_prob = corrupt_probs[target_id].item()
    corrupt_rank = (corrupt_probs > corrupt_probs[target_id]).sum().item() + 1

    # Initialize result matrices
    recovery_matrix = []
    prob_matrix = []

    # Sweep all 144 heads - patch one at a time
    for layer in range(N_LAYERS):
        layer_recovery = []
        layer_probs = []

        for head in range(N_HEADS):
            # Create patching hook - replace corrupt activation with clean
            hook_name = f"blocks.{layer}.attn.hook_z"
            clean_head_activation = clean_cache[hook_name][:, -1, head, :].clone()

            def make_patch_hook(clean_act):
                def hook_fn(value, hook):
                    value = value.clone()
                    value[:, -1, head, :] = clean_act
                    return value
                return hook_fn

            # Run corrupt prompt with this head patched
            patched_logits = model.run_with_hooks(
                corrupt_toks, fwd_hooks=[(hook_name, make_patch_hook(clean_head_activation))]
            )

            patched_probs = patched_logits[0, -1].softmax(-1)
            patched_prob = patched_probs[target_id].item()

            # Calculate recovery percentage
            if abs(clean_prob - corrupt_prob) > 1e-10:
                recovery = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob) * 100
            else:
                recovery = 0.0

            layer_recovery.append(recovery)
            layer_probs.append(patched_prob)

        recovery_matrix.append(layer_recovery)
        prob_matrix.append(layer_probs)

    # Get top-k tokens helper
    def get_top_k(probs, k=10):
        vals, idx = probs.topk(k)
        tokens = [model.to_string(i.item()) for i in idx]
        return [TokenProb(token=t, prob=p.item()) for t, p in zip(tokens, vals)]

    return PatchSweepResponse(
        clean_prompt=request.clean_prompt,
        corrupt_prompt=request.corrupt_prompt,
        target_token=request.target_token,
        clean_prob=clean_prob,
        corrupt_prob=corrupt_prob,
        clean_rank=clean_rank,
        corrupt_rank=corrupt_rank,
        recovery_matrix=recovery_matrix,
        prob_matrix=prob_matrix,
        top_clean=get_top_k(clean_probs),
        top_corrupt=get_top_k(corrupt_probs),
    )


@app.post("/sae-features", response_model=SAEResponse)
def sae_features(request: SAERequest):
    """Get SAE feature activations for input text."""
    tokens = model.to_tokens(request.text)

    # Run model and cache activations
    _, cache = model.run_with_cache(tokens)

    # Get activations at the SAE hook point
    activations = cache[SAE_HOOK_POINT]  # Shape: [batch, pos, d_model]

    # Encode through SAE to get feature activations
    # Shape: [batch, pos, n_features]
    feature_acts = sae.encode(activations)

    # Get token strings
    token_strs = model.to_str_tokens(tokens[0])

    # Build per-token feature info
    token_features_list = []
    all_feature_acts = []  # For finding top overall

    for pos, tok_str in enumerate(token_strs):
        pos_acts = feature_acts[0, pos]  # Shape: [n_features]

        # Get top-k features for this position
        top_vals, top_indices = pos_acts.topk(min(request.top_k, (pos_acts > 0).sum().item()))

        top_features = []
        for feat_idx, feat_val in zip(top_indices.tolist(), top_vals.tolist()):
            if feat_val > 0:  # Only include active features
                top_features.append(FeatureActivation(
                    feature_index=feat_idx,
                    activation=feat_val,
                    neuronpedia_url=f"https://neuronpedia.org/gpt2-small/8-res-jb/{feat_idx}"
                ))
                all_feature_acts.append((feat_idx, feat_val, pos))

        token_features_list.append(TokenFeatures(
            token=tok_str,
            position=pos,
            top_features=top_features
        ))

    # Get top features overall (across all positions)
    all_feature_acts.sort(key=lambda x: x[1], reverse=True)
    seen_features = set()
    top_overall = []
    for feat_idx, feat_val, pos in all_feature_acts:
        if feat_idx not in seen_features and len(top_overall) < request.top_k:
            top_overall.append(FeatureActivation(
                feature_index=feat_idx,
                activation=feat_val,
                neuronpedia_url=f"https://neuronpedia.org/gpt2-small/8-res-jb/{feat_idx}"
            ))
            seen_features.add(feat_idx)

    return SAEResponse(
        text=request.text,
        hook_point=SAE_HOOK_POINT,
        n_features=SAE_N_FEATURES,
        tokens=token_features_list,
        top_features_overall=top_overall,
    )


@app.post("/residual-sweep", response_model=ResidualSweepResponse)
def residual_sweep(request: ResidualSweepRequest):
    """
    Residual stream sweep: For each layer, patch its clean residual stream
    activation into the corrupt run and measure recovery of target token probability.

    This identifies which layers are critical for passing information about
    entities/variables through the model.
    """
    target_id = model.to_single_token(request.target_token)
    clean_toks = model.to_tokens(request.clean_prompt)
    corrupt_toks = model.to_tokens(request.corrupt_prompt)

    # Cache clean activations
    _, clean_cache = model.run_with_cache(clean_toks)

    # Get baseline probabilities
    clean_logits = model(clean_toks)
    clean_probs = clean_logits[0, -1].softmax(-1)
    clean_prob = clean_probs[target_id].item()
    clean_rank = (clean_probs > clean_probs[target_id]).sum().item() + 1

    corrupt_logits = model(corrupt_toks)
    corrupt_probs = corrupt_logits[0, -1].softmax(-1)
    corrupt_prob = corrupt_probs[target_id].item()
    corrupt_rank = (corrupt_probs > corrupt_probs[target_id]).sum().item() + 1

    # Initialize result arrays
    recovery_by_layer = []
    prob_by_layer = []

    # Sweep all 12 layers - patch residual stream one at a time
    for layer in range(N_LAYERS):
        # Patch the residual stream at this layer
        hook_name = f"blocks.{layer}.hook_resid_post"
        clean_resid = clean_cache[hook_name][:, -1, :].clone()

        def make_patch_hook(clean_act):
            def hook_fn(value, hook):
                value = value.clone()
                value[:, -1, :] = clean_act
                return value
            return hook_fn

        # Run corrupt prompt with this layer's residual stream patched
        patched_logits = model.run_with_hooks(
            corrupt_toks, fwd_hooks=[(hook_name, make_patch_hook(clean_resid))]
        )

        patched_probs = patched_logits[0, -1].softmax(-1)
        patched_prob = patched_probs[target_id].item()

        # Calculate recovery percentage
        if abs(clean_prob - corrupt_prob) > 1e-10:
            recovery = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob) * 100
        else:
            recovery = 0.0

        recovery_by_layer.append(recovery)
        prob_by_layer.append(patched_prob)

    # Get top-k tokens helper
    def get_top_k(probs, k=10):
        vals, idx = probs.topk(k)
        tokens = [model.to_string(i.item()) for i in idx]
        return [TokenProb(token=t, prob=p.item()) for t, p in zip(tokens, vals)]

    return ResidualSweepResponse(
        clean_prompt=request.clean_prompt,
        corrupt_prompt=request.corrupt_prompt,
        target_token=request.target_token,
        clean_prob=clean_prob,
        corrupt_prob=corrupt_prob,
        clean_rank=clean_rank,
        corrupt_rank=corrupt_rank,
        recovery_by_layer=recovery_by_layer,
        prob_by_layer=prob_by_layer,
        top_clean=get_top_k(clean_probs),
        top_corrupt=get_top_k(corrupt_probs),
    )


@app.post("/multi-residual-patch", response_model=MultiResidualPatchResponse)
def multi_residual_patch(request: MultiResidualPatchRequest):
    """Patch multiple layers' residual streams simultaneously and measure combined recovery."""
    target_id = model.to_single_token(request.target_token)
    clean_toks = model.to_tokens(request.clean_prompt)
    corrupt_toks = model.to_tokens(request.corrupt_prompt)

    # Cache clean activations
    _, clean_cache = model.run_with_cache(clean_toks)

    # Get baseline probabilities
    clean_logits = model(clean_toks)
    clean_probs = clean_logits[0, -1].softmax(-1)
    clean_prob = clean_probs[target_id].item()
    clean_rank = (clean_probs > clean_probs[target_id]).sum().item() + 1

    corrupt_logits = model(corrupt_toks)
    corrupt_probs = corrupt_logits[0, -1].softmax(-1)
    corrupt_prob = corrupt_probs[target_id].item()
    corrupt_rank = (corrupt_probs > corrupt_probs[target_id]).sum().item() + 1

    # Create hooks for all layers to patch
    hooks = []
    for layer_spec in request.layers:
        hook_name = f"blocks.{layer_spec.layer}.hook_resid_post"
        clean_resid = clean_cache[hook_name][:, -1, :].clone()

        def make_patch_hook(clean_act):
            def hook_fn(value, hook):
                value = value.clone()
                value[:, -1, :] = clean_act
                return value
            return hook_fn

        hooks.append((hook_name, make_patch_hook(clean_resid)))

    # Run corrupt prompt with all patches applied simultaneously
    if hooks:
        patched_logits = model.run_with_hooks(corrupt_toks, fwd_hooks=hooks)
    else:
        patched_logits = corrupt_logits

    patched_probs = patched_logits[0, -1].softmax(-1)
    patched_prob = patched_probs[target_id].item()
    patched_rank = (patched_probs > patched_probs[target_id]).sum().item() + 1

    # Calculate recovery percentage
    if abs(clean_prob - corrupt_prob) > 1e-10:
        recovery_pct = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob) * 100
    else:
        recovery_pct = 0.0

    # Get top-10 tokens
    def get_top_k(probs, k=10):
        vals, idx = probs.topk(k)
        tokens = [model.to_string(i.item()) for i in idx]
        return [TokenProb(token=t, prob=p.item()) for t, p in zip(tokens, vals)]

    return MultiResidualPatchResponse(
        clean_prompt=request.clean_prompt,
        corrupt_prompt=request.corrupt_prompt,
        target_token=request.target_token,
        layers=request.layers,
        clean_prob=clean_prob,
        corrupt_prob=corrupt_prob,
        patched_prob=patched_prob,
        clean_rank=clean_rank,
        corrupt_rank=corrupt_rank,
        patched_rank=patched_rank,
        recovery_pct=recovery_pct,
        top_clean=get_top_k(clean_probs),
        top_corrupt=get_top_k(corrupt_probs),
        top_patched=get_top_k(patched_probs),
    )


@app.post("/mlp-sweep", response_model=MLPSweepResponse)
def mlp_sweep(request: MLPSweepRequest):
    """
    MLP sweep: For each layer, patch its clean MLP output activation
    into the corrupt run and measure recovery of target token probability.

    This identifies which MLP layers are critical for the task.
    Hook point: blocks.{layer}.mlp.hook_post
    """
    target_id = model.to_single_token(request.target_token)
    clean_toks = model.to_tokens(request.clean_prompt)
    corrupt_toks = model.to_tokens(request.corrupt_prompt)

    # Cache clean activations
    _, clean_cache = model.run_with_cache(clean_toks)

    # Get baseline probabilities
    clean_logits = model(clean_toks)
    clean_probs = clean_logits[0, -1].softmax(-1)
    clean_prob = clean_probs[target_id].item()
    clean_rank = (clean_probs > clean_probs[target_id]).sum().item() + 1

    corrupt_logits = model(corrupt_toks)
    corrupt_probs = corrupt_logits[0, -1].softmax(-1)
    corrupt_prob = corrupt_probs[target_id].item()
    corrupt_rank = (corrupt_probs > corrupt_probs[target_id]).sum().item() + 1

    # Initialize result arrays
    recovery_by_layer = []
    prob_by_layer = []

    # Sweep all 12 layers - patch MLP output one at a time
    for layer in range(N_LAYERS):
        # Patch the MLP output at this layer
        hook_name = f"blocks.{layer}.mlp.hook_post"
        clean_mlp = clean_cache[hook_name][:, -1, :].clone()

        def make_patch_hook(clean_act):
            def hook_fn(value, hook):
                value = value.clone()
                value[:, -1, :] = clean_act
                return value
            return hook_fn

        # Run corrupt prompt with this layer's MLP output patched
        patched_logits = model.run_with_hooks(
            corrupt_toks, fwd_hooks=[(hook_name, make_patch_hook(clean_mlp))]
        )

        patched_probs = patched_logits[0, -1].softmax(-1)
        patched_prob = patched_probs[target_id].item()

        # Calculate recovery percentage
        if abs(clean_prob - corrupt_prob) > 1e-10:
            recovery = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob) * 100
        else:
            recovery = 0.0

        recovery_by_layer.append(recovery)
        prob_by_layer.append(patched_prob)

    # Get top-k tokens helper
    def get_top_k(probs, k=10):
        vals, idx = probs.topk(k)
        tokens = [model.to_string(i.item()) for i in idx]
        return [TokenProb(token=t, prob=p.item()) for t, p in zip(tokens, vals)]

    return MLPSweepResponse(
        clean_prompt=request.clean_prompt,
        corrupt_prompt=request.corrupt_prompt,
        target_token=request.target_token,
        clean_prob=clean_prob,
        corrupt_prob=corrupt_prob,
        clean_rank=clean_rank,
        corrupt_rank=corrupt_rank,
        recovery_by_layer=recovery_by_layer,
        prob_by_layer=prob_by_layer,
        top_clean=get_top_k(clean_probs),
        top_corrupt=get_top_k(corrupt_probs),
    )


@app.post("/multi-mlp-patch", response_model=MultiMLPPatchResponse)
def multi_mlp_patch(request: MultiMLPPatchRequest):
    """Patch multiple MLP layers simultaneously and measure combined recovery."""
    target_id = model.to_single_token(request.target_token)
    clean_toks = model.to_tokens(request.clean_prompt)
    corrupt_toks = model.to_tokens(request.corrupt_prompt)

    # Cache clean activations
    _, clean_cache = model.run_with_cache(clean_toks)

    # Get baseline probabilities
    clean_logits = model(clean_toks)
    clean_probs = clean_logits[0, -1].softmax(-1)
    clean_prob = clean_probs[target_id].item()
    clean_rank = (clean_probs > clean_probs[target_id]).sum().item() + 1

    corrupt_logits = model(corrupt_toks)
    corrupt_probs = corrupt_logits[0, -1].softmax(-1)
    corrupt_prob = corrupt_probs[target_id].item()
    corrupt_rank = (corrupt_probs > corrupt_probs[target_id]).sum().item() + 1

    # Create hooks for all MLP layers to patch
    hooks = []
    for layer_spec in request.layers:
        hook_name = f"blocks.{layer_spec.layer}.mlp.hook_post"
        clean_mlp = clean_cache[hook_name][:, -1, :].clone()

        def make_patch_hook(clean_act):
            def hook_fn(value, hook):
                value = value.clone()
                value[:, -1, :] = clean_act
                return value
            return hook_fn

        hooks.append((hook_name, make_patch_hook(clean_mlp)))

    # Run corrupt prompt with all patches applied simultaneously
    if hooks:
        patched_logits = model.run_with_hooks(corrupt_toks, fwd_hooks=hooks)
    else:
        patched_logits = corrupt_logits

    patched_probs = patched_logits[0, -1].softmax(-1)
    patched_prob = patched_probs[target_id].item()
    patched_rank = (patched_probs > patched_probs[target_id]).sum().item() + 1

    # Calculate recovery percentage
    if abs(clean_prob - corrupt_prob) > 1e-10:
        recovery_pct = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob) * 100
    else:
        recovery_pct = 0.0

    # Get top-10 tokens
    def get_top_k(probs, k=10):
        vals, idx = probs.topk(k)
        tokens = [model.to_string(i.item()) for i in idx]
        return [TokenProb(token=t, prob=p.item()) for t, p in zip(tokens, vals)]

    return MultiMLPPatchResponse(
        clean_prompt=request.clean_prompt,
        corrupt_prompt=request.corrupt_prompt,
        target_token=request.target_token,
        layers=request.layers,
        clean_prob=clean_prob,
        corrupt_prob=corrupt_prob,
        patched_prob=patched_prob,
        clean_rank=clean_rank,
        corrupt_rank=corrupt_rank,
        patched_rank=patched_rank,
        recovery_pct=recovery_pct,
        top_clean=get_top_k(clean_probs),
        top_corrupt=get_top_k(corrupt_probs),
        top_patched=get_top_k(patched_probs),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
