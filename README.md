# GPT-2 Activation Patching

Interactive web application for exploring activation patching in GPT-2 small, investigating how individual attention heads contribute to factual recall.

## Features

- **Activation Patching**: Patch individual attention heads from a clean run into a corrupted run
- **Recovery Metrics**: Measure how much patching recovers the target token probability
- **Rank Tracking**: Monitor how the target token rank changes (Clean → Corrupt → Patched)
- **Top-10 Tokens**: View side-by-side comparison of predicted tokens across clean, corrupt, and patched runs
- **Interactive Web UI**: Browser-based interface for experimenting with different prompts and heads
- **REST API**: FastAPI backend for programmatic access

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

1. Open http://localhost:8000
2. Enter your prompts:
   - **Clean prompt**: "Paris is the capital of"
   - **Corrupt prompt**: "Berlin is the capital of"
   - **Target token**: " France" (include leading space)
3. Select layer and head to patch (0-11)
4. Click "Run Patching"

### API Endpoint

**POST** `/patch`

Request:
```json
{
  "clean_prompt": "Paris is the capital of",
  "corrupt_prompt": "Berlin is the capital of",
  "target_token": " France",
  "layer": 8,
  "head": 11
}
```

Response:
```json
{
  "clean_prob": 0.3342,
  "corrupt_prob": 0.0032,
  "patched_prob": 0.0311,
  "recovery_pct": 8.4,
  "clean_rank": 1,
  "corrupt_rank": 523,
  "patched_rank": 42,
  "top_clean": [
    {"token": " France", "prob": 0.3342},
    {"token": " Paris", "prob": 0.1234},
    ...
  ],
  "top_corrupt": [
    {"token": " Germany", "prob": 0.5213},
    {"token": " the", "prob": 0.2190},
    ...
  ],
  "top_patched": [
    {"token": " Germany", "prob": 0.5224},
    {"token": " the", "prob": 0.2168},
    ...
  ]
}
```

## Scripts

### `src/phase1_patching_demo.py`
Sweeps all 144 attention heads (12 layers × 12 heads) and generates a heatmap showing which heads are important for factual recall.

```bash
python src/phase1_patching_demo.py
```

Outputs:
- Terminal table of recovery percentages
- `patching_heatmap.png` visualization
- Top-10 most important heads

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

**Activation Patching** is a technique to identify which components of a neural network are responsible for specific behaviors:

1. **Clean Run**: Run the model on the clean prompt ("Paris is the capital of") and cache all activations
2. **Corrupt Run**: Run on the corrupted prompt ("Berlin is the capital of")
3. **Patching**: Replace a single attention head's output in the corrupt run with the cached clean activation
4. **Measure Recovery**: See if the patched model now predicts the clean answer (" France")

If patching head `L8H11` significantly increases P(" France"), that head is important for the "Paris → France" fact.

## Project Structure

```
.
├── app.py                      # FastAPI web server
├── src/
│   ├── phase1_patching_demo.py # Full head sweep
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

- Python 3.11+
- PyTorch 2.6+
- transformer-lens
- FastAPI
- matplotlib (for heatmap generation)

## Tech Stack

- **Model**: GPT-2 small (124M parameters)
- **Framework**: TransformerLens for interpretability hooks
- **Backend**: FastAPI
- **Frontend**: Vanilla HTML/CSS/JS
- **Deployment**: Docker with optional GPU support

## Credits

Built for mechanistic interpretability research. Based on activation patching techniques from:
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/)

## License

MIT
