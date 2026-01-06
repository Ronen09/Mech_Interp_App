import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformer_lens import HookedTransformer
from sae_lens import SAE

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


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "n_layers": N_LAYERS, "n_heads": N_HEADS}


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>GPT-2 Circuit Analysis</title>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent: #58a6ff;
            --accent-hover: #79c0ff;
            --success: #3fb950;
            --warning: #d29922;
            --danger: #f85149;
            --gradient: linear-gradient(135deg, #58a6ff 0%, #a371f7 100%);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }

        /* Header */
        .header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 32px;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 600;
            background: var(--gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }

        /* Tabs */
        .tabs {
            display: flex;
            gap: 4px;
            margin-bottom: 24px;
            background: var(--bg-secondary);
            padding: 4px;
            border-radius: 12px;
            width: fit-content;
        }

        .tab {
            padding: 12px 24px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.2s ease;
        }

        .tab:hover {
            color: var(--text-primary);
            background: var(--bg-tertiary);
        }

        .tab.active {
            background: var(--accent);
            color: #fff;
        }

        /* Cards */
        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .card-subtitle {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 4px;
        }

        /* Form Elements */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-group.full-width {
            grid-column: 1 / -1;
        }

        label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        input, select {
            padding: 10px 14px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 14px;
            transition: all 0.2s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.15);
        }

        input::placeholder {
            color: var(--text-muted);
        }

        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 24px;
            font-size: 14px;
            font-weight: 500;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-primary {
            background: var(--accent);
            color: #fff;
        }

        .btn-primary:hover {
            background: var(--accent-hover);
        }

        .btn-secondary {
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border);
        }

        .btn-secondary:hover {
            background: var(--border);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .btn-group {
            display: flex;
            gap: 12px;
            margin-top: 20px;
        }

        /* Grid for head selection */
        .head-grid {
            display: grid;
            grid-template-columns: auto repeat(12, 1fr);
            gap: 3px;
            background: var(--bg-tertiary);
            padding: 16px;
            border-radius: 12px;
            font-size: 11px;
        }

        .head-grid-label {
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            font-weight: 500;
            padding: 4px;
        }

        .head-grid-cell {
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-secondary);
            border: 2px solid transparent;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
            font-size: 9px;
            color: var(--text-muted);
            position: relative;
        }

        .head-grid-cell:hover {
            border-color: var(--accent);
            transform: scale(1.1);
            z-index: 10;
        }

        .head-grid-cell.selected {
            border-color: var(--success);
            box-shadow: 0 0 8px rgba(63, 185, 80, 0.4);
        }

        .head-grid-cell.selected::after {
            content: '✓';
            position: absolute;
            font-size: 10px;
            color: var(--success);
        }

        .head-grid-cell .tooltip {
            display: none;
            position: absolute;
            bottom: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%);
            background: var(--bg-primary);
            border: 1px solid var(--border);
            padding: 8px 12px;
            border-radius: 8px;
            white-space: nowrap;
            font-size: 11px;
            z-index: 100;
            color: var(--text-primary);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }

        .head-grid-cell:hover .tooltip {
            display: block;
        }

        /* Results */
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .result-card {
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 16px;
        }

        .result-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 4px;
        }

        .result-value {
            font-size: 1.5rem;
            font-weight: 600;
            font-family: 'SF Mono', 'Consolas', monospace;
        }

        .result-value.positive { color: var(--success); }
        .result-value.negative { color: var(--danger); }
        .result-value.neutral { color: var(--warning); }

        /* Token Lists */
        .tokens-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
        }

        .token-list {
            background: var(--bg-tertiary);
            border-radius: 8px;
            overflow: hidden;
        }

        .token-list-header {
            padding: 12px 16px;
            background: var(--bg-primary);
            font-weight: 500;
            font-size: 0.875rem;
            border-bottom: 1px solid var(--border);
        }

        .token-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 16px;
            border-bottom: 1px solid var(--border);
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 13px;
            transition: background 0.15s ease;
        }

        .token-item:last-child {
            border-bottom: none;
        }

        .token-item:hover {
            background: var(--bg-secondary);
        }

        .token-item.clickable {
            cursor: pointer;
        }

        .token-item.selected {
            background: rgba(88, 166, 255, 0.1);
            border-left: 3px solid var(--accent);
        }

        .token-name {
            color: var(--text-primary);
        }

        .token-prob {
            color: var(--text-muted);
        }

        /* Selection info */
        .selection-info {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            background: rgba(88, 166, 255, 0.1);
            border: 1px solid rgba(88, 166, 255, 0.3);
            border-radius: 8px;
            margin-bottom: 16px;
        }

        .selection-count {
            font-weight: 600;
            color: var(--accent);
        }

        .selection-list {
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .clear-btn {
            margin-left: auto;
            padding: 6px 12px;
            font-size: 12px;
        }

        /* Tab content */
        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* Legend */
        .legend {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            margin-top: 16px;
            font-size: 12px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }

        /* Loading state */
        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Progress bar */
        .progress-bar {
            height: 8px;
            background: var(--bg-tertiary);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }

        .progress-fill {
            height: 100%;
            background: var(--gradient);
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        /* Section titles */
        .section-title {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }

        .section-title h3 {
            font-size: 1rem;
            font-weight: 600;
        }

        .badge {
            padding: 4px 10px;
            background: var(--accent);
            color: #fff;
            font-size: 11px;
            font-weight: 500;
            border-radius: 12px;
        }

        /* Split layout */
        .split-layout {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 24px;
        }

        @media (max-width: 1024px) {
            .split-layout {
                grid-template-columns: 1fr;
            }
        }

        /* Detail panel */
        .detail-panel {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            position: sticky;
            top: 24px;
        }

        .detail-row {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }

        .detail-row:last-child {
            border-bottom: none;
        }

        .metric-select {
            padding: 8px 12px;
            font-size: 13px;
            min-width: 200px;
        }

        /* Sweep top tokens */
        .sweep-tokens {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 8px;
        }

        @media (max-width: 768px) {
            .sweep-tokens {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>GPT-2 Circuit Analysis</h1>
            <p>Explore attention head contributions through activation patching and ablation</p>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('sweep')">Ablation Sweep</button>
            <button class="tab" onclick="switchTab('patchsweep')">Patch Sweep</button>
            <button class="tab" onclick="switchTab('sae')">SAE Features</button>
        </div>

        <!-- Sweep Tab -->
        <div id="tab-sweep" class="tab-content active">
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Full Head Sweep</div>
                        <div class="card-subtitle">Ablate all 144 heads and visualize their importance</div>
                    </div>
                    <select id="metricSelector" class="metric-select" onchange="updateGridColors()">
                        <option value="delta_probs">Δprob (probability drop)</option>
                        <option value="delta_margins">Δmargin (logit margin)</option>
                        <option value="delta_ranks">Δrank (rank degradation)</option>
                        <option value="delta_entropies">Δentropy (distribution flattening)</option>
                    </select>
                </div>

                <div class="form-grid" style="margin-bottom: 20px;">
                    <div class="form-group">
                        <label>Clean Prompt</label>
                        <input type="text" id="sweep_clean_prompt" value="Paris is the capital of">
                    </div>
                    <div class="form-group">
                        <label>Target Token</label>
                        <input type="text" id="sweep_target_token" value=" France">
                    </div>
                </div>

                <button class="btn btn-primary" onclick="runSweep()" id="sweepBtn">Run Full Sweep (144 heads)</button>
            </div>

            <div id="sweepResults" style="display: none;">
                <div class="card">
                    <div class="section-title">
                        <h3>Top-10 Tokens (Clean)</h3>
                    </div>
                    <div class="sweep-tokens" id="sweepTopClean"></div>
                </div>

                <div id="sweepSelectionInfo" class="selection-info" style="display: none;">
                    <span class="selection-count" id="sweepSelectionCountText">0 heads selected</span>
                    <span class="selection-list" id="sweepSelectionList"></span>
                    <button class="btn btn-secondary clear-btn" onclick="clearSweepSelection()">Clear All</button>
                </div>

                <div class="card">
                    <div class="section-title">
                        <h3>Attention Head Importance</h3>
                        <span style="color: var(--text-muted); font-size: 12px;">Click to select heads for multi-ablation</span>
                    </div>
                    <div class="head-grid" id="gridContainer"></div>
                    <div class="legend" id="gridLegend"></div>
                    <div class="btn-group" style="margin-top: 16px;">
                        <button class="btn btn-primary" onclick="runMultiAblate()" id="multiAblateBtn">Ablate Selected Heads</button>
                        <button class="btn btn-secondary" onclick="clearSweepSelection()">Clear Selection</button>
                    </div>
                </div>

                <div id="multiAblateResults" class="card" style="display: none;">
                    <div class="card-header">
                        <div class="card-title">Multi-Ablation Results</div>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px;">
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Clean</div>
                            <div class="detail-row"><span style="color: var(--text-muted);">P(target)</span><span id="ma_clean_prob" style="font-family: monospace;">-</span></div>
                            <div class="detail-row"><span style="color: var(--text-muted);">Rank</span><span id="ma_clean_rank" style="font-family: monospace;">-</span></div>
                        </div>
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Ablated</div>
                            <div class="detail-row"><span style="color: var(--text-muted);">P(target)</span><span id="ma_ablated_prob" style="font-family: monospace;">-</span></div>
                            <div class="detail-row"><span style="color: var(--text-muted);">Rank</span><span id="ma_ablated_rank" style="font-family: monospace;">-</span></div>
                            <div class="detail-row"><span style="color: var(--text-muted);">Δprob</span><span id="ma_delta_prob" style="font-family: monospace;">-</span></div>
                            <div class="detail-row"><span style="color: var(--text-muted);">Δmargin</span><span id="ma_delta_margin" style="font-family: monospace;">-</span></div>
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Top-5 Clean</div>
                            <div id="ma_top_clean"></div>
                        </div>
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Top-5 Ablated</div>
                            <div id="ma_top_ablated"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Patch Sweep Tab -->
        <div id="tab-patchsweep" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">Activation Patching Sweep</div>
                        <div class="card-subtitle">Patch clean activations into corrupt runs to find heads that store/pass variable bindings</div>
                    </div>
                    <select id="patchMetricSelector" class="metric-select" onchange="updatePatchGridColors()">
                        <option value="recovery">Recovery % (how much target prob is recovered)</option>
                        <option value="prob">Patched Probability (raw P(target))</option>
                    </select>
                </div>

                <div class="form-grid" style="margin-bottom: 20px;">
                    <div class="form-group">
                        <label>Clean Prompt</label>
                        <input type="text" id="patch_clean_prompt" value="John gave Mary the book. Mary gave it to">
                    </div>
                    <div class="form-group">
                        <label>Corrupt Prompt</label>
                        <input type="text" id="patch_corrupt_prompt" value="Bob gave Mary the book. Bob gave it to">
                    </div>
                    <div class="form-group">
                        <label>Target Token</label>
                        <input type="text" id="patch_target_token" value=" John">
                    </div>
                </div>

                <button class="btn btn-primary" onclick="runPatchSweep()" id="patchSweepBtn">Run Patch Sweep (144 heads)</button>
            </div>

            <div id="patchSweepResults" style="display: none;">
                <div class="card">
                    <div class="section-title">
                        <h3>Baseline Metrics</h3>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Clean</div>
                            <div class="detail-row"><span style="color: var(--text-muted);">P(target)</span><span id="patch_clean_prob" style="font-family: monospace;">-</span></div>
                            <div class="detail-row"><span style="color: var(--text-muted);">Rank</span><span id="patch_clean_rank" style="font-family: monospace;">-</span></div>
                        </div>
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Corrupt</div>
                            <div class="detail-row"><span style="color: var(--text-muted);">P(target)</span><span id="patch_corrupt_prob" style="font-family: monospace;">-</span></div>
                            <div class="detail-row"><span style="color: var(--text-muted);">Rank</span><span id="patch_corrupt_rank" style="font-family: monospace;">-</span></div>
                        </div>
                    </div>
                </div>

                <div id="patchSelectionInfo" class="selection-info" style="display: none;">
                    <span class="selection-count" id="patchSelectionCountText">0 heads selected</span>
                    <span class="selection-list" id="patchSelectionList"></span>
                    <button class="btn btn-secondary clear-btn" onclick="clearPatchSelection()">Clear All</button>
                </div>

                <div class="card">
                    <div class="section-title">
                        <h3>Head Recovery Heatmap</h3>
                        <span style="color: var(--text-muted); font-size: 12px;">Click to select heads for multi-patch</span>
                    </div>
                    <div class="head-grid" id="patchGridContainer"></div>
                    <div class="legend" id="patchGridLegend"></div>
                    <div class="btn-group" style="margin-top: 16px;">
                        <button class="btn btn-primary" onclick="runMultiPatch()" id="multiPatchBtn">Patch Selected Heads</button>
                        <button class="btn btn-secondary" onclick="clearPatchSelection()">Clear Selection</button>
                    </div>
                </div>

                <div id="multiPatchResults" class="card" style="display: none;">
                    <div class="card-header">
                        <div class="card-title">Multi-Patch Results</div>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px;">
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Clean</div>
                            <div class="detail-row"><span style="color: var(--text-muted);">P(target)</span><span id="mp_clean_prob" style="font-family: monospace;">-</span></div>
                            <div class="detail-row"><span style="color: var(--text-muted);">Rank</span><span id="mp_clean_rank" style="font-family: monospace;">-</span></div>
                        </div>
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Corrupt</div>
                            <div class="detail-row"><span style="color: var(--text-muted);">P(target)</span><span id="mp_corrupt_prob" style="font-family: monospace;">-</span></div>
                            <div class="detail-row"><span style="color: var(--text-muted);">Rank</span><span id="mp_corrupt_rank" style="font-family: monospace;">-</span></div>
                        </div>
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Patched</div>
                            <div class="detail-row"><span style="color: var(--text-muted);">P(target)</span><span id="mp_patched_prob" style="font-family: monospace;">-</span></div>
                            <div class="detail-row"><span style="color: var(--text-muted);">Rank</span><span id="mp_patched_rank" style="font-family: monospace;">-</span></div>
                            <div class="detail-row"><span style="color: var(--text-muted);">Recovery</span><span id="mp_recovery" style="font-family: monospace;">-</span></div>
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Top-5 Clean</div>
                            <div id="mp_top_clean"></div>
                        </div>
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Top-5 Corrupt</div>
                            <div id="mp_top_corrupt"></div>
                        </div>
                        <div>
                            <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Top-5 Patched</div>
                            <div id="mp_top_patched"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- SAE Features Tab -->
        <div id="tab-sae" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">SAE Feature Analysis</div>
                        <div class="card-subtitle">Decompose activations into interpretable features using Sparse Autoencoders</div>
                    </div>
                </div>

                <div class="form-grid" style="margin-bottom: 20px;">
                    <div class="form-group full-width">
                        <label>Input Text</label>
                        <textarea id="sae_text" rows="3" style="width: 100%; background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: 8px; padding: 12px; color: var(--text-primary); font-family: inherit; resize: vertical;">The Eiffel Tower is located in Paris, France. It was built in 1889.</textarea>
                    </div>
                    <div class="form-group">
                        <label>Top K Features per Token</label>
                        <input type="number" id="sae_top_k" value="10" min="1" max="50">
                    </div>
                </div>

                <button class="btn btn-primary" onclick="runSAEAnalysis()" id="saeBtn">Analyze Features</button>
            </div>

            <div id="saeResults" style="display: none;">
                <div class="card">
                    <div class="section-title">
                        <h3>SAE Info</h3>
                    </div>
                    <div class="detail-row"><span style="color: var(--text-muted);">Hook Point</span><span id="sae_info_hook" style="font-family: monospace;">-</span></div>
                    <div class="detail-row"><span style="color: var(--text-muted);">Total Features</span><span id="sae_info_nfeatures" style="font-family: monospace;">-</span></div>
                </div>

                <div class="card">
                    <div class="section-title">
                        <h3>Top Features (Overall)</h3>
                        <span style="color: var(--text-muted); font-size: 12px;">Click to view on Neuronpedia</span>
                    </div>
                    <div id="saeTopFeatures" style="max-height: 400px; overflow-y: auto;"></div>
                </div>

                <div class="card">
                    <div class="section-title">
                        <h3>Features by Token</h3>
                    </div>
                    <div id="saeTokenFeatures" style="max-height: 500px; overflow-y: auto;"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // ============== Global State ==============
        let sweepData = null;
        let selectedSweepClampToken = null;
        let circuitSelectedHeads = new Set();
        let patchSweepData = null;
        let patchSelectedHeads = new Set();

        // ============== Tab Switching ==============
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            document.querySelector(`.tab[onclick="switchTab('${tabName}')"]`).classList.add('active');
            document.getElementById(`tab-${tabName}`).classList.add('active');

            // Initialize circuit grid if switching to circuit tab
            if (tabName === 'circuit') {
                initCircuitGrid();
            }
        }

        // ============== Sweep Mode ==============
        async function runSweep() {
            const btn = document.getElementById('sweepBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Running sweep...';

            const data = {
                clean_prompt: document.getElementById('sweep_clean_prompt').value,
                target_token: document.getElementById('sweep_target_token').value,
                clamp_token: selectedSweepClampToken
            };

            try {
                const resp = await fetch('/sweep', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                sweepData = await resp.json();

                document.getElementById('sweepResults').style.display = 'block';
                renderSweepTokens();
                renderSweepGrid();
                updateGridColors();
                updateSweepClampBadge();
            } catch (e) {
                alert('Error: ' + e.message);
            }

            btn.disabled = false;
            btn.textContent = 'Run Full Sweep (144 heads)';
        }

        function renderSweepTokens() {
            if (!sweepData || !sweepData.top_clean) return;

            const container = document.getElementById('sweepTopClean');
            const html = sweepData.top_clean.map((t, i) => {
                const selectedClass = (selectedSweepClampToken === t.token) ? 'selected' : '';
                return `<div class="token-item clickable ${selectedClass}" onclick="selectSweepClampToken('${t.token.replace(/'/g, "\\'")}')">
                    <span class="token-name">${i+1}. "${t.token}"</span>
                    <span class="token-prob">${t.prob.toFixed(4)}</span>
                </div>`;
            }).join('');
            container.innerHTML = html;
        }

        async function selectSweepClampToken(token) {
            if (selectedSweepClampToken === token) {
                selectedSweepClampToken = null;
            } else {
                selectedSweepClampToken = token;
            }
            await runSweep();
        }

        function updateSweepClampBadge() {
            const badge = document.getElementById('sweepClampBadge');
            if (selectedSweepClampToken) {
                badge.textContent = `Clamped: "${selectedSweepClampToken}"`;
                badge.style.display = 'inline';
            } else {
                badge.style.display = 'none';
            }
        }

        function renderSweepGrid() {
            const container = document.getElementById('gridContainer');
            container.innerHTML = '';

            // Add header row
            const emptyCorner = document.createElement('div');
            emptyCorner.className = 'head-grid-label';
            container.appendChild(emptyCorner);

            for (let h = 0; h < 12; h++) {
                const label = document.createElement('div');
                label.className = 'head-grid-label';
                label.textContent = `H${h}`;
                container.appendChild(label);
            }

            // Add rows
            for (let layer = 0; layer < 12; layer++) {
                const rowLabel = document.createElement('div');
                rowLabel.className = 'head-grid-label';
                rowLabel.textContent = `L${layer}`;
                container.appendChild(rowLabel);

                for (let head = 0; head < 12; head++) {
                    const cell = document.createElement('div');
                    cell.className = 'head-grid-cell';
                    cell.dataset.layer = layer;
                    cell.dataset.head = head;

                    const tooltip = document.createElement('div');
                    tooltip.className = 'tooltip';
                    const dProb = sweepData.delta_probs[layer][head];
                    const dRank = sweepData.delta_ranks[layer][head];
                    const dEntropy = sweepData.delta_entropies[layer][head];
                    const dMargin = sweepData.delta_margins[layer][head];

                    tooltip.innerHTML = `<strong>L${layer}H${head}</strong><br>Δprob: ${dProb >= 0 ? '+' : ''}${dProb.toFixed(4)}<br>Δrank: ${dRank >= 0 ? '+' : ''}${Math.round(dRank)}<br>Δmargin: ${dMargin >= 0 ? '+' : ''}${dMargin.toFixed(3)}`;

                    cell.appendChild(tooltip);
                    cell.onclick = (e) => selectSweepGridCell(layer, head, e);
                    container.appendChild(cell);
                }
            }
        }

        function updateGridColors() {
            if (!sweepData) return;

            const metricName = document.getElementById('metricSelector').value;
            const metricData = sweepData[metricName];

            const allValues = metricData.flat();
            const minVal = Math.min(...allValues);
            const maxVal = Math.max(...allValues);

            document.getElementById('gridLegend').innerHTML = `
                <div class="legend-item"><div class="legend-color" style="background: hsl(220, 70%, 45%);"></div> Low impact</div>
                <div class="legend-item"><div class="legend-color" style="background: hsl(0, 70%, 55%);"></div> High impact</div>
                <span style="margin-left: auto; color: var(--text-muted);">Range: ${minVal.toFixed(3)} to ${maxVal.toFixed(3)}</span>
            `;

            const cells = document.querySelectorAll('#gridContainer .head-grid-cell');
            cells.forEach(cell => {
                const layer = parseInt(cell.dataset.layer);
                const head = parseInt(cell.dataset.head);
                const value = metricData[layer][head];

                const normalized = maxVal > minVal ? (value - minVal) / (maxVal - minVal) : 0.5;
                const hue = 220 - (normalized * 220);
                const saturation = 60 + (normalized * 20);
                const lightness = 35 + (normalized * 25);

                cell.style.backgroundColor = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
            });
        }

        async function selectSweepGridCell(layer, head, event) {
            document.querySelectorAll('#gridContainer .head-grid-cell').forEach(c => c.classList.remove('selected'));
            event.currentTarget.classList.add('selected');

            document.getElementById('layer').value = layer;
            document.getElementById('head').value = head;

            const panel = document.getElementById('detailPanel');
            const content = document.getElementById('detailContent');
            panel.style.display = 'block';

            content.innerHTML = '<p style="color: var(--text-muted);">Loading...</p>';

            try {
                const data = {
                    clean_prompt: document.getElementById('sweep_clean_prompt').value,
                    corrupt_prompt: document.getElementById('sweep_clean_prompt').value,
                    target_token: document.getElementById('sweep_target_token').value,
                    layer: layer,
                    head: head,
                    mode: 'ablate',
                    clamp_token: null
                };

                const resp = await fetch('/patch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await resp.json();

                content.innerHTML = `
                    <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 16px; color: var(--accent);">L${layer}H${head}</div>
                    <div class="detail-row"><span style="color: var(--text-muted);">Δprob</span><span style="font-family: monospace;">${result.delta_prob >= 0 ? '+' : ''}${result.delta_prob.toFixed(4)}</span></div>
                    <div class="detail-row"><span style="color: var(--text-muted);">Δrank</span><span style="font-family: monospace;">${result.delta_rank >= 0 ? '+' : ''}${result.delta_rank}</span></div>
                    <div class="detail-row"><span style="color: var(--text-muted);">Δentropy</span><span style="font-family: monospace;">${result.delta_entropy >= 0 ? '+' : ''}${result.delta_entropy.toFixed(3)}</span></div>
                    <div class="detail-row"><span style="color: var(--text-muted);">Clean P</span><span style="font-family: monospace;">${result.clean_prob.toFixed(4)}</span></div>
                    <div class="detail-row"><span style="color: var(--text-muted);">Ablated P</span><span style="font-family: monospace;">${result.patched_prob.toFixed(4)}</span></div>
                    <div style="margin-top: 16px;">
                        <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Top-5 Clean</div>
                        ${result.top_clean.slice(0, 5).map((t, i) => `<div class="detail-row" style="font-size: 12px;"><span>"${t.token}"</span><span>${t.prob.toFixed(4)}</span></div>`).join('')}
                    </div>
                    <div style="margin-top: 16px;">
                        <div style="font-weight: 500; margin-bottom: 8px; color: var(--text-secondary);">Top-5 Ablated</div>
                        ${result.top_patched.slice(0, 5).map((t, i) => `<div class="detail-row" style="font-size: 12px;"><span>"${t.token}"</span><span>${t.prob.toFixed(4)}</span></div>`).join('')}
                    </div>
                `;
            } catch (e) {
                content.innerHTML = `<p style="color: var(--danger);">Error: ${e.message}</p>`;
            }
        }

        // ============== Circuit Analysis ==============
        function initCircuitGrid() {
            const container = document.getElementById('circuitGrid');
            if (container.children.length > 0) return; // Already initialized

            container.innerHTML = '';

            // Add header row
            const emptyCorner = document.createElement('div');
            emptyCorner.className = 'head-grid-label';
            container.appendChild(emptyCorner);

            for (let h = 0; h < 12; h++) {
                const label = document.createElement('div');
                label.className = 'head-grid-label';
                label.textContent = `H${h}`;
                container.appendChild(label);
            }

            // Add rows
            for (let layer = 0; layer < 12; layer++) {
                const rowLabel = document.createElement('div');
                rowLabel.className = 'head-grid-label';
                rowLabel.textContent = `L${layer}`;
                container.appendChild(rowLabel);

                for (let head = 0; head < 12; head++) {
                    const cell = document.createElement('div');
                    cell.className = 'head-grid-cell';
                    cell.dataset.layer = layer;
                    cell.dataset.head = head;
                    cell.style.backgroundColor = 'var(--bg-secondary)';

                    const tooltip = document.createElement('div');
                    tooltip.className = 'tooltip';
                    tooltip.textContent = `L${layer}H${head} - Click to select`;
                    cell.appendChild(tooltip);

                    cell.onclick = () => toggleCircuitHead(layer, head, cell);
                    container.appendChild(cell);
                }
            }
        }

        function toggleCircuitHead(layer, head, cell) {
            const key = `${layer}-${head}`;

            if (circuitSelectedHeads.has(key)) {
                circuitSelectedHeads.delete(key);
                cell.classList.remove('selected');
            } else {
                circuitSelectedHeads.add(key);
                cell.classList.add('selected');
            }

            updateCircuitSelectionUI();
        }

        function updateCircuitSelectionUI() {
            const count = circuitSelectedHeads.size;
            document.getElementById('selectedCount').textContent = `${count} selected`;

            const selectionInfo = document.getElementById('circuitSelectionInfo');
            if (count > 0) {
                selectionInfo.style.display = 'flex';
                document.getElementById('selectionCountText').textContent = `${count} head${count > 1 ? 's' : ''} selected`;

                const headsList = Array.from(circuitSelectedHeads).map(k => {
                    const [l, h] = k.split('-');
                    return `L${l}H${h}`;
                }).join(', ');
                document.getElementById('selectionList').textContent = headsList;
            } else {
                selectionInfo.style.display = 'none';
            }
        }

        function clearCircuitSelection() {
            circuitSelectedHeads.clear();
            document.querySelectorAll('#circuitGrid .head-grid-cell').forEach(c => c.classList.remove('selected'));
            updateCircuitSelectionUI();
            document.getElementById('circuitResults').style.display = 'none';
        }

        async function runCircuitAnalysis() {
            if (circuitSelectedHeads.size === 0) {
                alert('Please select at least one head to ablate');
                return;
            }

            const btn = document.getElementById('circuitBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Analyzing...';

            const heads = Array.from(circuitSelectedHeads).map(k => {
                const [layer, head] = k.split('-').map(Number);
                return { layer, head };
            });

            const data = {
                clean_prompt: document.getElementById('circuit_clean_prompt').value,
                target_token: document.getElementById('circuit_target_token').value,
                heads: heads
            };

            try {
                const resp = await fetch('/multi-ablate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await resp.json();

                document.getElementById('circuitResults').style.display = 'block';

                document.getElementById('circuit_clean_prob').textContent = result.clean_prob.toFixed(4);
                document.getElementById('circuit_ablated_prob').textContent = result.ablated_prob.toFixed(4);
                document.getElementById('circuit_delta_prob').textContent = (result.delta_prob >= 0 ? '-' : '+') + Math.abs(result.delta_prob).toFixed(4);
                document.getElementById('circuit_delta_prob').className = 'result-value ' + (result.delta_prob > 0 ? 'negative' : 'positive');
                document.getElementById('circuit_delta_rank').textContent = (result.delta_rank >= 0 ? '+' : '') + result.delta_rank;
                document.getElementById('circuit_delta_rank').className = 'result-value ' + (result.delta_rank > 0 ? 'negative' : 'positive');

                document.getElementById('circuit_clean_rank').textContent = '#' + result.clean_rank;
                document.getElementById('circuit_ablated_rank').textContent = '#' + result.ablated_rank;
                document.getElementById('circuit_delta_entropy').textContent = (result.delta_entropy >= 0 ? '+' : '') + result.delta_entropy.toFixed(3);
                document.getElementById('circuit_delta_margin').textContent = (result.delta_margin >= 0 ? '+' : '') + result.delta_margin.toFixed(3);

                renderCircuitTokens(result.top_clean, 'circuit_top_clean');
                renderCircuitTokens(result.top_ablated, 'circuit_top_ablated');
            } catch (e) {
                alert('Error: ' + e.message);
            }

            btn.disabled = false;
            btn.textContent = 'Analyze Circuit';
        }

        function renderCircuitTokens(tokens, elementId) {
            const html = tokens.map((t, i) => `
                <div class="token-item">
                    <span class="token-name">${i+1}. "${t.token}"</span>
                    <span class="token-prob">${t.prob.toFixed(4)}</span>
                </div>
            `).join('');
            document.getElementById(elementId).innerHTML = html;
        }

        // Initialize circuit grid on page load
        document.addEventListener('DOMContentLoaded', () => {
            // Pre-initialize the circuit grid
            setTimeout(initCircuitGrid, 100);
        });

        // ============== Patch Sweep Functions ==============
        async function runPatchSweep() {
            const btn = document.getElementById('patchSweepBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Running patch sweep...';

            const data = {
                clean_prompt: document.getElementById('patch_clean_prompt').value,
                corrupt_prompt: document.getElementById('patch_corrupt_prompt').value,
                target_token: document.getElementById('patch_target_token').value
            };

            try {
                const resp = await fetch('/patch-sweep', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                patchSweepData = await resp.json();

                document.getElementById('patchSweepResults').style.display = 'block';

                // Display baseline metrics
                document.getElementById('patch_clean_prob').textContent = patchSweepData.clean_prob.toFixed(4);
                document.getElementById('patch_corrupt_prob').textContent = patchSweepData.corrupt_prob.toFixed(4);
                document.getElementById('patch_clean_rank').textContent = patchSweepData.clean_rank;
                document.getElementById('patch_corrupt_rank').textContent = patchSweepData.corrupt_rank;

                renderPatchGrid();
                updatePatchGridColors();
            } catch (e) {
                alert('Error: ' + e.message);
            }

            btn.disabled = false;
            btn.textContent = 'Run Patch Sweep (144 heads)';
        }

        function renderPatchGrid() {
            const container = document.getElementById('patchGridContainer');
            container.innerHTML = '';

            // Add header row
            const emptyCorner = document.createElement('div');
            emptyCorner.className = 'head-grid-label';
            container.appendChild(emptyCorner);

            for (let h = 0; h < 12; h++) {
                const label = document.createElement('div');
                label.className = 'head-grid-label';
                label.textContent = `H${h}`;
                container.appendChild(label);
            }

            // Add rows
            for (let layer = 0; layer < 12; layer++) {
                const rowLabel = document.createElement('div');
                rowLabel.className = 'head-grid-label';
                rowLabel.textContent = `L${layer}`;
                container.appendChild(rowLabel);

                for (let head = 0; head < 12; head++) {
                    const cell = document.createElement('div');
                    cell.className = 'head-grid-cell';
                    cell.dataset.layer = layer;
                    cell.dataset.head = head;

                    const tooltip = document.createElement('div');
                    tooltip.className = 'tooltip';
                    const recovery = patchSweepData.recovery_matrix[layer][head];
                    const prob = patchSweepData.prob_matrix[layer][head];

                    tooltip.innerHTML = `<strong>L${layer}H${head}</strong><br>Recovery: ${recovery.toFixed(1)}%<br>P(target): ${prob.toFixed(4)}`;

                    cell.appendChild(tooltip);
                    cell.onclick = (e) => togglePatchHead(layer, head, e);
                    container.appendChild(cell);
                }
            }
        }

        function togglePatchHead(layer, head, event) {
            const key = `${layer}-${head}`;
            const cell = event.currentTarget;

            if (patchSelectedHeads.has(key)) {
                patchSelectedHeads.delete(key);
                cell.classList.remove('selected');
            } else {
                patchSelectedHeads.add(key);
                cell.classList.add('selected');
            }

            updatePatchSelectionUI();
        }

        function updatePatchSelectionUI() {
            const count = patchSelectedHeads.size;
            const infoBar = document.getElementById('patchSelectionInfo');
            const countText = document.getElementById('patchSelectionCountText');
            const listText = document.getElementById('patchSelectionList');

            if (count > 0) {
                infoBar.style.display = 'flex';
                countText.textContent = `${count} head${count > 1 ? 's' : ''} selected`;

                const sorted = Array.from(patchSelectedHeads).sort((a, b) => {
                    const [la, ha] = a.split('-').map(Number);
                    const [lb, hb] = b.split('-').map(Number);
                    return la - lb || ha - hb;
                });
                listText.textContent = sorted.map(k => {
                    const [l, h] = k.split('-');
                    return `L${l}H${h}`;
                }).join(', ');
            } else {
                infoBar.style.display = 'none';
            }
        }

        function clearPatchSelection() {
            patchSelectedHeads.clear();
            document.querySelectorAll('#patchGridContainer .head-grid-cell').forEach(c => c.classList.remove('selected'));
            updatePatchSelectionUI();
            document.getElementById('multiPatchResults').style.display = 'none';
        }

        async function runMultiPatch() {
            if (patchSelectedHeads.size === 0) {
                alert('Please select at least one head to patch');
                return;
            }

            const btn = document.getElementById('multiPatchBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Patching...';

            const heads = Array.from(patchSelectedHeads).map(k => {
                const [layer, head] = k.split('-').map(Number);
                return { layer, head };
            });

            const data = {
                clean_prompt: document.getElementById('patch_clean_prompt').value,
                corrupt_prompt: document.getElementById('patch_corrupt_prompt').value,
                target_token: document.getElementById('patch_target_token').value,
                heads: heads
            };

            try {
                const resp = await fetch('/multi-patch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await resp.json();

                // Display results
                document.getElementById('mp_clean_prob').textContent = result.clean_prob.toFixed(4);
                document.getElementById('mp_corrupt_prob').textContent = result.corrupt_prob.toFixed(4);
                document.getElementById('mp_patched_prob').textContent = result.patched_prob.toFixed(4);
                document.getElementById('mp_clean_rank').textContent = result.clean_rank;
                document.getElementById('mp_corrupt_rank').textContent = result.corrupt_rank;
                document.getElementById('mp_patched_rank').textContent = result.patched_rank;
                document.getElementById('mp_recovery').textContent = result.recovery_pct.toFixed(1) + '%';

                const renderTokens = (tokens, elementId) => {
                    document.getElementById(elementId).innerHTML = tokens.slice(0, 5).map((t, i) =>
                        `<div class="detail-row" style="font-size: 12px;"><span>"${t.token}"</span><span>${t.prob.toFixed(4)}</span></div>`
                    ).join('');
                };

                renderTokens(result.top_clean, 'mp_top_clean');
                renderTokens(result.top_corrupt, 'mp_top_corrupt');
                renderTokens(result.top_patched, 'mp_top_patched');

                document.getElementById('multiPatchResults').style.display = 'block';
            } catch (e) {
                alert('Error: ' + e.message);
            }

            btn.disabled = false;
            btn.textContent = 'Patch Selected Heads';
        }

        function updatePatchGridColors() {
            if (!patchSweepData) return;

            const metric = document.getElementById('patchMetricSelector').value;
            const metricData = metric === 'recovery' ? patchSweepData.recovery_matrix : patchSweepData.prob_matrix;

            const allValues = metricData.flat();
            const minVal = Math.min(...allValues);
            const maxVal = Math.max(...allValues);

            document.getElementById('patchGridLegend').innerHTML = `
                <div class="legend-item"><div class="legend-color" style="background: hsl(220, 70%, 45%);"></div> Low ${metric === 'recovery' ? 'recovery' : 'prob'}</div>
                <div class="legend-item"><div class="legend-color" style="background: hsl(0, 70%, 55%);"></div> High ${metric === 'recovery' ? 'recovery' : 'prob'}</div>
                <span style="margin-left: auto; color: var(--text-muted);">Range: ${minVal.toFixed(3)} to ${maxVal.toFixed(3)}</span>
            `;

            const cells = document.querySelectorAll('#patchGridContainer .head-grid-cell');
            cells.forEach(cell => {
                const layer = parseInt(cell.dataset.layer);
                const head = parseInt(cell.dataset.head);
                const value = metricData[layer][head];

                const normalized = maxVal > minVal ? (value - minVal) / (maxVal - minVal) : 0.5;
                const hue = 220 - (normalized * 220);
                const saturation = 60 + (normalized * 20);
                const lightness = 35 + (normalized * 25);

                cell.style.backgroundColor = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
            });
        }

        // ============== SAE Feature Analysis ==============
        async function runSAEAnalysis() {
            const btn = document.getElementById('saeBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Analyzing features...';

            const data = {
                text: document.getElementById('sae_text').value,
                top_k: parseInt(document.getElementById('sae_top_k').value)
            };

            try {
                const resp = await fetch('/sae-features', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await resp.json();

                // Update info
                document.getElementById('sae_info_hook').textContent = result.hook_point;
                document.getElementById('sae_info_nfeatures').textContent = result.n_features.toLocaleString();

                // Render top features overall
                renderSAETopFeatures(result.top_features_overall);

                // Render features by token
                renderSAETokenFeatures(result.tokens);

                document.getElementById('saeResults').style.display = 'block';
            } catch (e) {
                alert('Error: ' + e.message);
            }

            btn.disabled = false;
            btn.textContent = 'Analyze Features';
        }

        function renderSAETopFeatures(features) {
            const container = document.getElementById('saeTopFeatures');

            const html = features.map((f, i) => `
                <div class="detail-row" style="cursor: pointer;" onclick="window.open('${f.neuronpedia_url}', '_blank')">
                    <span>
                        <span style="color: var(--accent); font-weight: 500;">#${f.feature_index}</span>
                        <span style="color: var(--text-muted); margin-left: 8px;">→ Neuronpedia</span>
                    </span>
                    <span style="font-family: monospace; color: var(--success);">${f.activation.toFixed(4)}</span>
                </div>
            `).join('');

            container.innerHTML = html || '<p style="color: var(--text-muted);">No features activated</p>';
        }

        function renderSAETokenFeatures(tokens) {
            const container = document.getElementById('saeTokenFeatures');

            const html = tokens.map((t, i) => {
                const escapedToken = t.token.replace(/</g, '&lt;').replace(/>/g, '&gt;');
                const featuresList = t.top_features.length > 0
                    ? t.top_features.slice(0, 5).map(f =>
                        `<span class="feature-badge" onclick="window.open('${f.neuronpedia_url}', '_blank')" style="
                            display: inline-block;
                            background: var(--bg-tertiary);
                            border: 1px solid var(--border);
                            padding: 2px 8px;
                            margin: 2px;
                            border-radius: 4px;
                            font-size: 11px;
                            cursor: pointer;
                        " title="Activation: ${f.activation.toFixed(4)}">
                            #${f.feature_index} <span style="color: var(--success);">(${f.activation.toFixed(2)})</span>
                        </span>`
                    ).join('')
                    : '<span style="color: var(--text-muted); font-size: 12px;">No features</span>';

                return `
                    <div style="margin-bottom: 12px; padding: 8px; background: var(--bg-secondary); border-radius: 6px;">
                        <div style="margin-bottom: 6px;">
                            <span style="color: var(--text-muted); font-size: 11px;">[${t.position}]</span>
                            <span style="font-family: monospace; background: var(--bg-tertiary); padding: 2px 6px; border-radius: 4px;">${escapedToken}</span>
                        </div>
                        <div>${featuresList}</div>
                    </div>
                `;
            }).join('');

            container.innerHTML = html;
        }

    </script>
</body>
</html>
"""


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
