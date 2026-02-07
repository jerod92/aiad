# aiad — AI-Assisted Drawing

Train a neural network to trace raster images using CAD-style drawing tools.

**Two-phase training pipeline:**
1. **Supervised pre-training** — Generate random shapes (primitives, polygons, splines, compounds, scenes), simulate mid-drawing states, train the model to predict the next correct cursor position and action.
2. **PPO fine-tuning** — The model draws entire images in a simulated CAD environment, rewarded by thick-line IoU between its output and the target.

## Architecture

```
Input (6ch): [target_RGB | drawing | ghost_preview | cursor]
  → UNet Encoder (4 stages, base_channels → base_channels×8)
    → Transformer Bridge (state injection: tool + click → N-layer transformer)
      → UNet Decoder (skip connections → full resolution)
        → Marginal X/Y position heads (factored softmax)
        → Tool / Click / Snap / End / Value heads (from pooled bridge features)
```

### Model Presets

| Preset | Resolution | Base channels | Transformer | Use case |
|--------|-----------|---------------|-------------|----------|
| `large` | 512×512 | 32 | 2-layer, 4-head | Serious training (multi-GPU) |
| `mini` | 256×256 | 16 | 1-layer, 2-head | Kaggle / PoC / debugging (~16GB T4) |

### Tools (24-slot vector)

**Active tools (0–7):** None, Line, Circle, Rectangle, Arc, Ellipse, RegPolygon, Spline

**Reserved slots (8–23):** Move, Rotate, Scale, Mirror, Offset, Trim, Fillet, Chamfer, Hatch, Dimension, Array, ConstrLine, Text, Select, Extend, Undo

The tool vector is intentionally oversized so future tools can be added without retraining from scratch.

### Shape Generators (30 total)

| Category | Generators |
|----------|-----------|
| **Primitives** (7) | line, polyline, polygon, circle, rectangle, arc, ellipse |
| **Polygons** (6) | regular polygon, pentagon, hexagon, 5-point star, 6-point star, 8-point star |
| **Splines** (5) | quadratic bezier, cubic bezier, S-curve, closed bezier, random spline |
| **Compounds** (7) | lattice, triangle mesh, hex mesh, concentric circles, concentric polygons, radial pattern, multi-shape |
| **Scenes** (5) | building floorplan, park, mechanical part, garden design, building elevation |

## Quick Start

```bash
git clone https://github.com/jerod92/aiad.git
cd aiad
pip install -e ".[dev]"
```

### Supervised Pre-training

```bash
# Large model (512px)
python -m aiad.train_supervised --epochs 5 --batch-size 16 --num-samples 50000

# Mini model (256px, Kaggle-friendly)
python -m aiad.train_supervised --epochs 5 --batch-size 32 --num-samples 50000 --model-size mini
```

### PPO Fine-tuning

```bash
python -m aiad.train_ppo --pretrained checkpoints/supervised/best.pt --num-episodes 2000

# Mini model
python -m aiad.train_ppo --pretrained checkpoints/supervised/best.pt --num-episodes 2000 --model-size mini
```

### Interactive Demo

Trace an uploaded image or a random shape with a trained model:

```bash
# Trace a user-provided image
python -m aiad.demo --checkpoint checkpoints/ppo/best.pt --image photo.png

# Trace a random synthetic shape
python -m aiad.demo --checkpoint checkpoints/ppo/best.pt

# Mini model
python -m aiad.demo --checkpoint checkpoints/ppo/best.pt --model-size mini
```

Output: an animated GIF showing target | overlay | canvas side-by-side.

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

### Jupyter Notebook

Paste this in a notebook cell:

```python
import subprocess, os

# Clone and install
subprocess.run(["git", "clone", "https://github.com/jerod92/aiad.git"], check=True)
os.chdir("aiad")
subprocess.run(["pip", "install", "-e", "."], check=True)

# --- Supervised pre-training (mini for Kaggle T4) ---
from aiad.train_supervised import train as sl_train
model = sl_train(epochs=3, batch_size=32, num_samples=100_000, model_size="mini")

# --- PPO fine-tuning ---
from aiad.train_ppo import train as ppo_train
model = ppo_train(pretrained="checkpoints/supervised/best.pt",
                  num_episodes=1000, model_size="mini")
```

### Bash Terminal (Vast.ai SSH / JupyterLab terminal)

```bash
git clone https://github.com/jerod92/aiad.git
cd aiad
pip install -e .
python -m aiad.train_supervised --epochs 3 --batch-size 32 --num-samples 100000 --model-size mini
python -m aiad.train_ppo --pretrained checkpoints/supervised/best.pt --num-episodes 1000 --model-size mini
```

> **Note:** Do not use Jupyter magic commands (`!git`, `%cd`, `!pip`) in a bash
> shell — they cause `bash: !git: event not found` errors.  Use the Python
> `subprocess` approach above for notebook cells, or plain commands in a
> terminal.

Both training functions return the model and save checkpoints automatically.
Download your best model: `checkpoints/ppo/best.pt` (or `supervised/best.pt`).

### Resuming Training

```python
# Resume from where you left off
model = sl_train(resume="checkpoints/supervised/latest.pt", epochs=2, model_size="mini")
model = ppo_train(resume="checkpoints/ppo/latest.pt", num_episodes=500, model_size="mini")
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
model = CADModel.from_preset("mini")
metadata = load_checkpoint("checkpoints/ppo/best.pt", model)
print(metadata)  # {'avg_iou': 0.73, 'episode': 1200, ...}
```

## Project Structure

```
aiad/
├── aiad/
│   ├── config.py             # Tools (24-slot), presets, device
│   ├── model.py              # UNet + Transformer + action heads
│   ├── dataset.py            # Synthetic shape dataset (supervised)
│   ├── env.py                # CAD environment (PPO) with thick-line IoU
│   ├── raster.py             # Anti-aliased OpenCV drawing utilities
│   ├── checkpoint.py         # Save/load/best-model tracking
│   ├── viz.py                # Visualization toolkit (CLI + library)
│   ├── demo.py               # Interactive tracing demo (GIF output)
│   ├── train_supervised.py   # Supervised pre-training
│   ├── train_ppo.py          # PPO fine-tuning
│   └── shapes/               # Shape generation package
│       ├── primitives.py     # Lines, circles, rectangles, arcs, ellipses
│       ├── polygons.py       # Regular polygons, stars
│       ├── splines.py        # Bezier curves (quad, cubic, random)
│       ├── compounds.py      # Lattices, meshes, concentric patterns
│       └── scenes.py         # Floorplans, parks, mechanical parts
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

**Anti-aliased rendering:** All raster primitives use `cv2.LINE_AA` for smooth, convolution-friendly visual input. This helps the UNet encoder extract clean features without aliasing artifacts.

**Thick-line IoU reward:** Both the target and the model's drawing are dilated before computing IoU. This means a line drawn parallel and close to the target gets partial credit rather than zero overlap. Controlled by `--reward-thickness` (default 10px).

**Factored X/Y prediction:** Instead of a full H×W heatmap, the model predicts marginal distributions over X (columns) and Y (rows) independently. This is memory-efficient and works well in practice.

**Ghost preview layer:** The environment provides a "ghost" channel showing what the current in-progress shape would look like (e.g., a line from the last anchor point to the cursor). This gives the model spatial context about its active tool state.

**Spline anchor points:** For Bezier curves, the model must learn to place control points that are *not visible* in the target image — the target shows the smooth curve, but the actions are off-curve control points. This is a key learning challenge.

**24-slot tool vector:** The tool selection head outputs logits for all 24 tool slots. During training, only 8 are active; the model learns to suppress the reserved slots. When new tools are added, training can continue from existing weights.

**Graceful interrupts:** Both training scripts handle Ctrl+C by saving a checkpoint before exiting, so you never lose progress.
