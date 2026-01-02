import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

app = FastAPI(title="GPT-2 Activation Patching")

# Load model at startup
print("Loading gpt2-small...")
model = HookedTransformer.from_pretrained("gpt2-small")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

N_LAYERS = model.cfg.n_layers
N_HEADS = model.cfg.n_heads


class PatchRequest(BaseModel):
    clean_prompt: str = "Paris is the capital of"
    corrupt_prompt: str = "Berlin is the capital of"
    target_token: str = " France"
    layer: int = 8
    head: int = 11
    mode: str = "patch"  # "patch" or "ablate"
    clamp_token: str | None = None  # For ablation: token to clamp to baseline


class TokenProb(BaseModel):
    token: str
    prob: float


class PatchResponse(BaseModel):
    clean_prompt: str
    corrupt_prompt: str
    target_token: str
    layer: int
    head: int
    clean_prob: float
    corrupt_prob: float
    patched_prob: float
    recovery_pct: float
    clean_rank: int
    corrupt_rank: int
    patched_rank: int
    top_clean: list[TokenProb]
    top_corrupt: list[TokenProb]
    top_patched: list[TokenProb]
    delta_prob: float = 0.0
    delta_rank: int = 0
    delta_entropy: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "n_layers": N_LAYERS, "n_heads": N_HEADS}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>GPT-2 Activation Patching</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d4ff; }
        .form-group {
            margin: 15px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #aaa;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #333;
            border-radius: 5px;
            background: #16213e;
            color: #eee;
            font-size: 14px;
        }
        .row {
            display: flex;
            gap: 15px;
        }
        .row .form-group {
            flex: 1;
        }
        button {
            background: #00d4ff;
            color: #1a1a2e;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover {
            background: #00a8cc;
        }
        button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        #results {
            margin-top: 30px;
            padding: 20px;
            background: #16213e;
            border-radius: 10px;
            display: none;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #333;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-label { color: #aaa; }
        .metric-value {
            font-weight: bold;
            font-family: monospace;
            font-size: 16px;
        }
        .recovery {
            font-size: 24px;
            color: #00d4ff;
            text-align: center;
            margin-top: 20px;
        }
        .recovery-bar {
            height: 20px;
            background: #333;
            border-radius: 10px;
            margin-top: 10px;
            overflow: hidden;
        }
        .recovery-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        .tokens-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .token-list {
            background: #0f1624;
            padding: 15px;
            border-radius: 8px;
        }
        .token-list h3 {
            margin: 0 0 10px 0;
            color: #00d4ff;
            font-size: 16px;
        }
        .token-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #1a2332;
            font-family: monospace;
            font-size: 13px;
        }
        .token-item:last-child {
            border-bottom: none;
        }
        .token-name {
            color: #eee;
            white-space: pre;
        }
        .token-prob {
            color: #888;
        }
        .token-item.clickable {
            cursor: pointer;
            transition: background 0.2s;
        }
        .token-item.clickable:hover {
            background: #1a2332;
        }
        .token-item.selected {
            background: #2a4a5a;
            border-left: 3px solid #00d4ff;
        }
    </style>
</head>
<body>
    <h1>GPT-2 Activation Patching & Ablation</h1>
    <p>Patch or ablate a single attention head and measure the impact on predictions.</p>

    <div class="form-group">
        <label>Clean Prompt</label>
        <input type="text" id="clean_prompt" value="Paris is the capital of">
    </div>

    <div class="form-group">
        <label>Corrupt Prompt</label>
        <input type="text" id="corrupt_prompt" value="Berlin is the capital of">
    </div>

    <div class="form-group">
        <label>Target Token (include leading space if needed)</label>
        <input type="text" id="target_token" value=" France">
    </div>

    <div class="row">
        <div class="form-group">
            <label>Layer (0-11)</label>
            <input type="number" id="layer" value="8" min="0" max="11">
        </div>
        <div class="form-group">
            <label>Head (0-11)</label>
            <input type="number" id="head" value="11" min="0" max="11">
        </div>
    </div>

    <div class="form-group">
        <label>Mode</label>
        <select id="mode" onchange="handleModeChange()">
            <option value="patch">Patch (replace with clean)</option>
            <option value="ablate">Ablate (zero out)</option>
        </select>
    </div>

    <button onclick="runPatch()" id="patchBtn">Run</button>

    <div id="results">
        <div class="metric">
            <span class="metric-label">Clean P(target)</span>
            <span class="metric-value" id="clean_prob">-</span>
        </div>
        <div class="metric">
            <span class="metric-label">Corrupt P(target)</span>
            <span class="metric-value" id="corrupt_prob">-</span>
        </div>
        <div class="metric">
            <span class="metric-label" id="modified_label">Patched P(target)</span>
            <span class="metric-value" id="patched_prob">-</span>
        </div>
        <div class="metric">
            <span class="metric-label" id="rank_label">Target Rank (Clean → Corrupt → Patched)</span>
            <span class="metric-value" id="ranks">-</span>
        </div>
        <div class="recovery">
            <div><span id="recovery_label">Recovery</span>: <span id="recovery_pct">-</span></div>
            <div class="recovery-bar">
                <div class="recovery-fill" id="recovery_bar" style="width: 0%"></div>
            </div>
        </div>

        <div class="tokens-grid">
            <div class="token-list">
                <h3 id="clean_tokens_label">Top-10 Tokens (Clean)</h3>
                <div id="top_clean"></div>
            </div>
            <div class="token-list">
                <h3>Top-10 Tokens (Corrupt)</h3>
                <div id="top_corrupt"></div>
            </div>
            <div class="token-list">
                <h3 id="modified_tokens_label">Top-10 Tokens (Patched)</h3>
                <div id="top_patched"></div>
            </div>
        </div>
    </div>

    <script>
        let selectedClampToken = null;
        let lastRequestData = null;

        function handleModeChange() {
            // Clear selection when switching modes
            selectedClampToken = null;
            document.querySelectorAll('#top_clean .token-item').forEach(el => {
                el.classList.remove('selected');
            });
        }

        async function selectTokenForClamping(token) {
            // Toggle selection - if clicking the same token, unselect it
            if (selectedClampToken === token) {
                selectedClampToken = null;
            } else {
                selectedClampToken = token;
            }

            // Update visual selection (will be reflected in next render)
            // Note: Visual update happens automatically when runPatch re-renders

            // Automatically re-run with the selected token if we're in ablate mode
            if (lastRequestData && lastRequestData.mode === 'ablate') {
                await runPatch();
            }
        }

        async function runPatch() {
            const btn = document.getElementById('patchBtn');
            btn.disabled = true;
            btn.textContent = 'Running...';

            const mode = document.getElementById('mode').value;
            const data = {
                clean_prompt: document.getElementById('clean_prompt').value,
                corrupt_prompt: document.getElementById('corrupt_prompt').value,
                target_token: document.getElementById('target_token').value,
                layer: parseInt(document.getElementById('layer').value),
                head: parseInt(document.getElementById('head').value),
                mode: mode,
                clamp_token: mode === 'ablate' ? selectedClampToken : null
            };

            // Store for reactive updates
            lastRequestData = data;

            try {
                const resp = await fetch('/patch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await resp.json();
                const mode = data.mode;
                const modeText = mode === 'patch' ? 'Patched' : 'Ablated';

                document.getElementById('results').style.display = 'block';

                if (mode === 'patch') {
                    // Show all metrics for patch mode
                    document.getElementById('modified_label').textContent = 'Patched P(target)';
                    document.getElementById('rank_label').textContent = 'Target Rank (Clean → Corrupt → Patched)';
                    document.getElementById('modified_tokens_label').textContent = 'Top-10 Tokens (Patched)';
                    document.getElementById('clean_tokens_label').textContent = 'Top-10 Tokens (Clean)';

                    document.querySelectorAll('.metric').forEach(el => el.style.display = 'flex');
                    document.querySelector('.recovery').style.display = 'block';

                    document.getElementById('clean_prob').textContent = result.clean_prob.toFixed(4);
                    document.getElementById('corrupt_prob').textContent = result.corrupt_prob.toFixed(4);
                    document.getElementById('patched_prob').textContent = result.patched_prob.toFixed(4);
                    document.getElementById('ranks').textContent = `#${result.clean_rank} → #${result.corrupt_rank} → #${result.patched_rank}`;
                    document.getElementById('recovery_label').textContent = 'Recovery';
                    document.getElementById('recovery_pct').textContent = result.recovery_pct.toFixed(1) + '%';
                    document.getElementById('recovery_bar').style.width = Math.min(100, Math.max(0, result.recovery_pct)) + '%';

                    // Show all 3 token columns
                    document.querySelectorAll('.token-list').forEach(el => el.style.display = 'block');
                    document.querySelector('.tokens-grid').style.gridTemplateColumns = '1fr 1fr 1fr';
                } else {
                    // Show specific metrics for ablation mode
                    const clampSuffix = data.clamp_token ? ` (+ "${data.clamp_token}" clamped)` : '';
                    document.getElementById('modified_label').textContent = 'Clean + ablation P(target)' + clampSuffix;
                    document.getElementById('rank_label').textContent = 'Clean degradation';
                    document.getElementById('modified_tokens_label').textContent = 'Top-10 Tokens (Ablated' + clampSuffix + ')';
                    document.getElementById('clean_tokens_label').textContent = 'Top-10 Tokens (Clean) - Click to clamp (auto-updates)';

                    // Show only first 3 metrics and hide recovery
                    const metrics = document.querySelectorAll('.metric');
                    metrics[0].style.display = 'flex';  // Clean P(target)
                    metrics[1].style.display = 'none';  // Corrupt P(target) - hide
                    metrics[2].style.display = 'flex';  // Ablated P(target)
                    metrics[3].style.display = 'flex';  // Degradation
                    document.querySelector('.recovery').style.display = 'none';

                    document.getElementById('clean_prob').textContent = result.clean_prob.toFixed(4);
                    document.getElementById('patched_prob').textContent = result.patched_prob.toFixed(4);
                    document.getElementById('ranks').textContent = `Δprob=${result.delta_prob.toFixed(4)}, Δrank=${result.delta_rank > 0 ? '+' : ''}${result.delta_rank}, Δentropy=${result.delta_entropy.toFixed(3)}`;

                    // Show only 2 token columns (clean and ablated)
                    document.querySelectorAll('.token-list')[0].style.display = 'block'; // Clean
                    document.querySelectorAll('.token-list')[1].style.display = 'none';  // Corrupt (hide)
                    document.querySelectorAll('.token-list')[2].style.display = 'block'; // Ablated
                    document.querySelector('.tokens-grid').style.gridTemplateColumns = '1fr 1fr';
                }

                // Display top tokens
                function renderTokens(tokens, elementId, clickable = false) {
                    const html = tokens.map((t, i) => {
                        const clickClass = clickable ? 'clickable' : '';
                        const selectedClass = (clickable && selectedClampToken === t.token) ? 'selected' : '';
                        const clickAttr = clickable ? `onclick="selectTokenForClamping('${t.token.replace(/'/g, "\\'")}')"` : '';
                        return `<div class="token-item ${clickClass} ${selectedClass}" ${clickAttr}>
                            <span class="token-name">${i+1}. "${t.token}"</span>
                            <span class="token-prob">${t.prob.toFixed(4)}</span>
                        </div>`;
                    }).join('');
                    document.getElementById(elementId).innerHTML = html;
                }

                // Make clean tokens clickable only in ablation mode
                const isAblateMode = mode === 'ablate';
                renderTokens(result.top_clean, 'top_clean', isAblateMode);
                renderTokens(result.top_corrupt, 'top_corrupt');
                renderTokens(result.top_patched, 'top_patched');
            } catch (e) {
                alert('Error: ' + e.message);
            }

            btn.disabled = false;
            btn.textContent = 'Run';
        }
    </script>
</body>
</html>
"""


@app.post("/patch", response_model=PatchResponse)
def patch(request: PatchRequest):
    target_id = model.to_single_token(request.target_token)

    # Tokenize
    clean_toks = model.to_tokens(request.clean_prompt)
    corrupt_toks = model.to_tokens(request.corrupt_prompt)

    # Cache clean activations
    _, clean_cache = model.run_with_cache(clean_toks)

    # Get baseline probabilities
    clean_logits = model(clean_toks)
    clean_probs = clean_logits[0, -1].softmax(-1)
    clean_prob = clean_probs[target_id].item()

    corrupt_logits = model(corrupt_toks)
    corrupt_probs = corrupt_logits[0, -1].softmax(-1)
    corrupt_prob = corrupt_probs[target_id].item()

    # Patch or ablate the head
    hook_name = f"blocks.{request.layer}.attn.hook_z"

    if request.mode == "patch":
        clean_head = clean_cache[hook_name][:, -1, request.head, :].clone()
        def hook_fn(value, hook):
            value = value.clone()
            value[:, -1, request.head, :] = clean_head
            return value

        patched_logits = model.run_with_hooks(
            corrupt_toks, fwd_hooks=[(hook_name, hook_fn)]
        )
        patched_probs = patched_logits[0, -1].softmax(-1)
        patched_prob = patched_probs[target_id].item()

        # Calculate recovery
        if clean_prob - corrupt_prob > 1e-10:
            recovery_pct = (patched_prob - corrupt_prob) / (clean_prob - corrupt_prob) * 100
        else:
            recovery_pct = 0.0
    else:  # ablate
        def hook_fn(value, hook):
            value = value.clone()
            value[:, -1, request.head, :] = 0.0
            return value

        # Run ablation on clean prompt
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
                pass  # If token is invalid, just skip clamping

        patched_probs = ablated_logits[0, -1].softmax(-1)
        patched_prob = patched_probs[target_id].item()

        # Store delta metrics (we'll send these in the response for ablation mode)
        recovery_pct = 0.0  # Not used for ablation

    # Calculate ranks (1-indexed)
    clean_rank = (clean_probs > clean_probs[target_id]).sum().item() + 1
    corrupt_rank = (corrupt_probs > corrupt_probs[target_id]).sum().item() + 1
    patched_rank = (patched_probs > patched_probs[target_id]).sum().item() + 1

    # Calculate deltas for ablation mode
    if request.mode == "ablate":
        # Δprob = p_clean - p_ablated
        delta_prob = clean_prob - patched_prob
        # Δrank = rank_ablated - rank_clean (positive means worse)
        delta_rank = patched_rank - clean_rank
        # Δentropy = H(p_ablated) - H(p_clean)
        clean_entropy = -(clean_probs * torch.log(clean_probs + 1e-12)).sum().item()
        ablated_entropy = -(patched_probs * torch.log(patched_probs + 1e-12)).sum().item()
        delta_entropy = ablated_entropy - clean_entropy
    else:
        delta_prob = 0.0
        delta_rank = 0
        delta_entropy = 0.0

    # Get top-10 tokens
    def get_top_k(probs, k=10):
        vals, idx = probs.topk(k)
        tokens = [model.to_string(i.item()) for i in idx]
        return [TokenProb(token=t, prob=p.item()) for t, p in zip(tokens, vals)]

    top_clean = get_top_k(clean_probs)
    top_corrupt = get_top_k(corrupt_probs)
    top_patched = get_top_k(patched_probs)

    return PatchResponse(
        clean_prompt=request.clean_prompt,
        corrupt_prompt=request.corrupt_prompt,
        target_token=request.target_token,
        layer=request.layer,
        head=request.head,
        clean_prob=clean_prob,
        corrupt_prob=corrupt_prob,
        patched_prob=patched_prob,
        recovery_pct=recovery_pct,
        clean_rank=clean_rank,
        corrupt_rank=corrupt_rank,
        patched_rank=patched_rank,
        top_clean=top_clean,
        top_corrupt=top_corrupt,
        top_patched=top_patched,
        delta_prob=delta_prob,
        delta_rank=delta_rank,
        delta_entropy=delta_entropy,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
