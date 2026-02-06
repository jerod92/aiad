# aiad — AI-Assisted Drawing

Train a neural network to trace raster images using CAD-style drawing tools (Line, Circle, etc.).

**Two-phase training pipeline:**
1. **Supervised pre-training** — Generate random shapes, simulate mid-drawing states, train the model to predict the next correct cursor position and action.
2. **PPO fine-tuning** — The model draws entire images in a simulated CAD environment, rewarded by thick-line IoU between its output and the target.

## Architecture

```
Input (6ch): [target_RGB | drawing | ghost_preview | cursor]
  → UNet Encoder (4 stages, 32→256 channels)
    → Transformer Bridge (state injection: tool + click → 2-layer transformer)
      → UNet Decoder (skip connections → full resolution)
        → Marginal X/Y position heads (factored softmax)
        → Tool / Click / Snap / End / Value heads (from pooled bridge features)
```

## Quick Start

```bash
git clone https://github.com/jerod92/aiad.git
cd aiad
pip install -e ".[dev]"
```

### Supervised Pre-training

```bash
python -m aiad.train_supervised --epochs 5 --batch-size 16 --num-samples 50000
```

### PPO Fine-tuning

```bash
python -m aiad.train_ppo --pretrained checkpoints/supervised/best.pt --num-episodes 2000
```

### Visualization (anytime)

```bash
# Inspect dataset quality
python -m aiad.viz dataset --num-samples 16

# Watch model draw (from a checkpoint)
python -m aiad.viz episode --checkpoint checkpoints/ppo/best.pt

# Plot training curves
python -m aiad.viz curves --checkpoint-dir checkpoints/supervised

# Compare predictions vs ground truth
python -m aiad.viz predict --checkpoint checkpoints/supervised/best.pt

# Dataset statistics (shape/action distributions)
python -m aiad.viz stats --num-samples 1000
```

## Kaggle / Vast.ai Quickstart

Paste this at the top of a notebook cell:

```python
!git clone https://github.com/jerod92/aiad.git
%cd aiad
!pip install -e .

# --- Supervised pre-training ---
from aiad.train_supervised import train as sl_train
model = sl_train(epochs=3, batch_size=32, num_samples=100_000)

# --- PPO fine-tuning ---
from aiad.train_ppo import train as ppo_train
model = ppo_train(pretrained="checkpoints/supervised/best.pt", num_episodes=1000)
```

Both training functions return the model and save checkpoints automatically.
Download your best model: `checkpoints/ppo/best.pt` (or `supervised/best.pt`).

### Resuming Training

```python
# Resume from where you left off
model = sl_train(resume="checkpoints/supervised/latest.pt", epochs=2)
model = ppo_train(resume="checkpoints/ppo/latest.pt", num_episodes=500)
```

## Checkpoint Management

Checkpoints are saved automatically during training:
- `checkpoints/<phase>/latest.pt` — most recent state (always saved)
- `checkpoints/<phase>/best.pt` — best model by loss (SL) or IoU (PPO)
- `checkpoints/<phase>/history.json` — full training log for curve plotting

Load a checkpoint manually:

```python
from aiad.checkpoint import load_checkpoint
from aiad.model import CADModel
model = CADModel()
metadata = load_checkpoint("checkpoints/ppo/best.pt", model)
print(metadata)  # {'avg_iou': 0.73, 'episode': 1200, ...}
```

## Project Structure

```
aiad/
├── aiad/
│   ├── config.py             # Tools, image size, device
│   ├── model.py              # UNet + Transformer + action heads
│   ├── dataset.py            # Synthetic shape dataset (supervised)
│   ├── env.py                # CAD environment (PPO) with thick-line IoU
│   ├── raster.py             # OpenCV drawing utilities
│   ├── checkpoint.py         # Save/load/best-model tracking
│   ├── viz.py                # Visualization toolkit (CLI + library)
│   ├── train_supervised.py   # Supervised pre-training
│   └── train_ppo.py          # PPO fine-tuning
├── tests/                    # pytest suite
├── outputs/                  # (gitignored) images, GIFs, stats
├── checkpoints/              # (gitignored) saved models
├── requirements.txt
└── pyproject.toml
```

## Running Tests

```bash
pytest tests/ -v
```

## Design Notes

**Thick-line IoU reward:** Both the target and the model's drawing are dilated before computing IoU. This means a line drawn parallel and close to the target gets partial credit rather than zero overlap. Controlled by `--reward-thickness` (default 10px).

**Factored X/Y prediction:** Instead of a full HxW heatmap, the model predicts marginal distributions over X (columns) and Y (rows) independently. This is memory-efficient and works well in practice.

**Ghost preview layer:** The environment provides a "ghost" channel showing what the current in-progress shape would look like (e.g., a line from the last anchor point to the cursor). This gives the model spatial context about its active tool state.

**Graceful interrupts:** Both training scripts handle Ctrl+C by saving a checkpoint before exiting, so you never lose progress.
