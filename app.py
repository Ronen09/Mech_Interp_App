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
    top_corrupt: list[TokenProb]
    top_patched: list[TokenProb]


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
    </style>
</head>
<body>
    <h1>GPT-2 Activation Patching</h1>
    <p>Patch a single attention head from clean to corrupt prompt and measure recovery.</p>

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

    <button onclick="runPatch()" id="patchBtn">Run Patching</button>

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
            <span class="metric-label">Patched P(target)</span>
            <span class="metric-value" id="patched_prob">-</span>
        </div>
        <div class="metric">
            <span class="metric-label">Target Rank (Clean → Corrupt → Patched)</span>
            <span class="metric-value" id="ranks">-</span>
        </div>
        <div class="recovery">
            <div>Recovery: <span id="recovery_pct">-</span></div>
            <div class="recovery-bar">
                <div class="recovery-fill" id="recovery_bar" style="width: 0%"></div>
            </div>
        </div>
    </div>

    <script>
        async function runPatch() {
            const btn = document.getElementById('patchBtn');
            btn.disabled = true;
            btn.textContent = 'Running...';

            const data = {
                clean_prompt: document.getElementById('clean_prompt').value,
                corrupt_prompt: document.getElementById('corrupt_prompt').value,
                target_token: document.getElementById('target_token').value,
                layer: parseInt(document.getElementById('layer').value),
                head: parseInt(document.getElementById('head').value)
            };

            try {
                const resp = await fetch('/patch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await resp.json();

                document.getElementById('results').style.display = 'block';
                document.getElementById('clean_prob').textContent = result.clean_prob.toFixed(4);
                document.getElementById('corrupt_prob').textContent = result.corrupt_prob.toFixed(4);
                document.getElementById('patched_prob').textContent = result.patched_prob.toFixed(4);
                document.getElementById('ranks').textContent = `#${result.clean_rank} → #${result.corrupt_rank} → #${result.patched_rank}`;
                document.getElementById('recovery_pct').textContent = result.recovery_pct.toFixed(1) + '%';
                document.getElementById('recovery_bar').style.width = Math.min(100, Math.max(0, result.recovery_pct)) + '%';
            } catch (e) {
                alert('Error: ' + e.message);
            }

            btn.disabled = false;
            btn.textContent = 'Run Patching';
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

    # Patch the head
    hook_name = f"blocks.{request.layer}.attn.hook_z"
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

    # Calculate ranks (1-indexed)
    clean_rank = (clean_probs > clean_probs[target_id]).sum().item() + 1
    corrupt_rank = (corrupt_probs > corrupt_probs[target_id]).sum().item() + 1
    patched_rank = (patched_probs > patched_probs[target_id]).sum().item() + 1

    # Get top-10 tokens
    def get_top_k(probs, k=10):
        vals, idx = probs.topk(k)
        tokens = [model.to_string(i.item()) for i in idx]
        return [TokenProb(token=t, prob=p.item()) for t, p in zip(tokens, vals)]

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
        top_corrupt=top_corrupt,
        top_patched=top_patched,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
