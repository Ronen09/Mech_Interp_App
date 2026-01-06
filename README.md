# GPT-2 Circuit Analysis

Interactive web application for exploring activation patching and ablation in GPT-2 small, investigating how attention heads and SAE features contribute to predictions.

## Features

- **Ablation Sweep**: Ablate all 144 heads and visualize Δprob, Δrank, Δentropy, and Δmargin heatmaps
- **Multi-Ablation**: Select heads in the grid and ablate them together for combined impact
- **Patch Sweep**: Patch clean activations into corrupt runs and visualize recovery or raw P(target)
- **Multi-Patch**: Patch selected heads simultaneously and measure recovery
- **SAE Feature Analysis**: Inspect top sparse autoencoder features per token with Neuronpedia links
- **Interactive Web UI**: Three tabs (Ablation Sweep, Patch Sweep, SAE Features)
- **REST API**: FastAPI endpoints for sweeps, multi-head ops, and SAE analysis

## Quick Start

### Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the web app
python app.py
```

Open http://localhost:8000 in your browser.

### Docker

```bash
# Build and run
docker compose up --build

# Or with GPU support
docker compose -f docker-compose.gpu.yml up --build
```

## Usage

### Web Interface

The UI has three tabs:

#### Ablation Sweep Tab
1. Enter clean prompt and target token
2. Click "Run Full Sweep (144 heads)"
3. View the heatmap showing head importance
4. Click heads to select them for multi-ablation
5. Use the metric dropdown to switch between Δprob, Δmargin, Δrank, Δentropy
6. Run "Ablate Selected Heads" for combined impact

#### Patch Sweep Tab
1. Enter clean prompt, corrupt prompt, and target token
2. Click "Run Patch Sweep (144 heads)"
3. View recovery or patched-prob heatmap
4. Select heads to run multi-patch
5. Compare clean/corrupt/patched metrics and top tokens

#### SAE Features Tab
1. Enter input text and top-k features per token
2. Click "Analyze Features"
3. Inspect top features overall and per token (click-through to Neuronpedia)

### API Endpoints

**GET** `/health`

Returns model/device info.

**POST** `/sweep`

Ablate all heads and return delta metrics.

Request:
```json
{
  "clean_prompt": "Paris is the capital of",
  "target_token": " France",
  "clamp_token": null
}
```

Response (abridged):
```json
{
  "delta_probs": [[0.0, 0.01, ...]],
  "delta_ranks": [[0.0, 2.0, ...]],
  "delta_entropies": [[0.0, 0.03, ...]],
  "delta_margins": [[0.0, 0.12, ...]],
  "clean_prob": 0.3342,
  "clean_rank": 1,
  "top_clean": [{"token": " France", "prob": 0.3342}, ...]
}
```

**POST** `/multi-ablate`

Ablate multiple heads simultaneously.

Request:
```json
{
  "clean_prompt": "Paris is the capital of",
  "target_token": " France",
  "heads": [
    {"layer": 8, "head": 11},
    {"layer": 9, "head": 6},
    {"layer": 10, "head": 0}
  ]
}
```

Response:
```json
{
  "clean_prob": 0.3342,
  "ablated_prob": 0.0891,
  "clean_rank": 1,
  "ablated_rank": 5,
  "delta_prob": 0.2451,
  "delta_rank": 4,
  "delta_entropy": 0.234,
  "delta_margin": 1.523,
  "top_clean": [{"token": " France", "prob": 0.3342}, ...],
  "top_ablated": [{"token": " the", "prob": 0.1523}, ...]
}
```

**POST** `/patch-sweep`

Patch clean activations into corrupt runs for all heads.

Request:
```json
{
  "clean_prompt": "John gave Mary the book. Mary gave it to",
  "corrupt_prompt": "Bob gave Mary the book. Bob gave it to",
  "target_token": " John"
}
```

Response (abridged):
```json
{
  "clean_prob": 0.2541,
  "corrupt_prob": 0.0124,
  "recovery_matrix": [[0.0, 3.1, ...]],
  "prob_matrix": [[0.02, 0.03, ...]],
  "top_clean": [{"token": " John", "prob": 0.2541}, ...],
  "top_corrupt": [{"token": " Bob", "prob": 0.4012}, ...]
}
```

**POST** `/multi-patch`

Patch multiple heads simultaneously for combined recovery.

Request:
```json
{
  "clean_prompt": "John gave Mary the book. Mary gave it to",
  "corrupt_prompt": "Bob gave Mary the book. Bob gave it to",
  "target_token": " John",
  "heads": [
    {"layer": 8, "head": 11},
    {"layer": 9, "head": 6}
  ]
}
```

Response (abridged):
```json
{
  "patched_prob": 0.0311,
  "recovery_pct": 8.4,
  "patched_rank": 42,
  "top_patched": [{"token": " John", "prob": 0.0311}, ...]
}
```

**POST** `/sae-features`

Return top SAE features per token and overall.

Request:
```json
{
  "text": "The Eiffel Tower is located in Paris, France",
  "top_k": 10
}
```

Response (abridged):
```json
{
  "hook_point": "blocks.8.hook_resid_pre",
  "n_features": 24576,
  "tokens": [{"token": "Paris", "position": 6, "top_features": [{"feature_index": 123, "activation": 3.21, "neuronpedia_url": "..."}]}],
  "top_features_overall": [{"feature_index": 456, "activation": 4.12, "neuronpedia_url": "..."}]
}
```

## Scripts

### `src/phase1_patching_demo.py`
Sweeps all 144 attention heads by patching clean activations into a corrupt run and generates recovery/probability heatmaps.

```bash
python src/phase1_patching_demo.py
```

Outputs:
- Terminal table of recovery percentages
- `patching_heatmap.png` visualization
- Top-10 most important heads

### `src/ablation_sweep.py`
Ablates all 144 heads on a clean prompt and prints delta metrics (prob, rank, entropy, margin) with optional heatmaps.

```bash
python src/ablation_sweep.py
```

### `src/patch_single_head.py`
Patches a single head and prints recovery metrics.

```bash
python src/patch_single_head.py
```

### `src/load_gpt2.py`
Simple script to load and test GPT-2 small.

```bash
python src/load_gpt2.py
```

## How It Works

### Activation Patching
A technique to identify which components are responsible for specific behaviors:

1. **Clean Run**: Run the model on the clean prompt and cache all activations
2. **Corrupt Run**: Run on the corrupted prompt
3. **Patching**: Replace a single attention head's output in the corrupt run with the cached clean activation
4. **Measure Recovery**: See if the patched model now predicts the clean answer

If patching head `L8H11` significantly increases P(" France"), that head is important for the "Paris → France" fact.

### Ablation
Zero out a head's output to measure its importance:
- **Single Head**: Ablate one head and measure Δprob, Δrank, Δentropy, Δmargin
- **Full Sweep**: Ablate all 144 heads individually to create importance heatmaps
- **Multi-Ablation**: Ablate selected heads together to measure combined impact

### SAE Feature Analysis
Decompose residual stream activations into sparse features:
- Uses `sae-lens` for GPT-2 small layer 8 residual stream
- Returns top features per token with links to Neuronpedia

## Project Structure

```
.
├── app.py                      # FastAPI web server
├── src/
│   ├── ablation_sweep.py       # Full ablation sweep
│   ├── phase1_patching_demo.py # Full patching sweep
│   ├── patch_single_head.py    # Single head patching
│   └── load_gpt2.py            # Model loader
├── Dockerfile
├── docker-compose.yml          # CPU version
├── docker-compose.gpu.yml      # GPU version
└── requirements.txt
```

## Example Results

**Paris → France Patching (L8H11)**
- Clean P(" France"): 33.4% (rank #1)
- Corrupt P(" France"): 0.3% (rank #523)
- Patched P(" France"): 3.1% (rank #42)
- **Recovery: 8.4%**

This shows L8H11 partially carries the "Paris → France" association, moving the target from rank 523 to rank 42.

## Requirements

- Python 3.11+ (Docker uses 3.11-slim)
- PyTorch
- transformer-lens
- sae-lens
- FastAPI + uvicorn
- matplotlib (for optional heatmap generation)

## Tech Stack

- **Model**: GPT-2 small (124M parameters)
- **Framework**: TransformerLens for interpretability hooks
- **SAE**: SAE-Lens for sparse autoencoder features
- **Backend**: FastAPI
- **Frontend**: Vanilla HTML/CSS/JS
- **Deployment**: Docker with optional GPU support

## Credits

Built for mechanistic interpretability research. Based on activation patching techniques from:
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/)

## License

MIT
