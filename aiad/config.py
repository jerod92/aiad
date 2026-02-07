"""
Global configuration: tool registry, model presets, image sizes.

The tool list is intentionally oversized (24 slots) so future tools can be
added without retraining from scratch.  Only indices 0-7 are *active* today;
the rest are reserved placeholders whose logits the model still learns to
suppress during training.
"""

import torch
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tool registry (24 slots — 8 active, 16 reserved)
# ---------------------------------------------------------------------------

TOOLS = [
    # --- Active tools (0-7) ---
    "None",         # 0  idle / tool deselect
    "Line",         # 1  polyline drawing (click anchors, snap to close)
    "Circle",       # 2  center + radius-point
    "Rectangle",    # 3  two-corner axis-aligned rectangle
    "Arc",          # 4  three-point arc (start, mid, end)
    "Ellipse",      # 5  center + semi-major + semi-minor
    "RegPolygon",   # 6  regular polygon (center + vertex, num_sides encoded)
    "Spline",       # 7  cubic Bezier (place control points, snap to finish)
    # --- Reserved / future tools (8-23) ---
    "Move",         # 8
    "Rotate",       # 9
    "Scale",        # 10
    "Mirror",       # 11
    "Offset",       # 12
    "Trim",         # 13
    "Fillet",       # 14
    "Chamfer",      # 15
    "Hatch",        # 16
    "Dimension",    # 17
    "Array",        # 18
    "ConstrLine",   # 19
    "Text",         # 20
    "Select",       # 21
    "Extend",       # 22
    "Undo",         # 23
]

TOOL_MAP = {name: i for i, name in enumerate(TOOLS)}
NUM_TOOLS = len(TOOLS)                     # 24
ACTIVE_TOOLS = set(range(8))               # indices 0-7 are trainable today

# ---------------------------------------------------------------------------
# Image size
# ---------------------------------------------------------------------------

IMG_SIZE = 512          # default (large) resolution — overridden by presets

# ---------------------------------------------------------------------------
# Model presets
# ---------------------------------------------------------------------------

@dataclass
class ModelPreset:
    name: str
    img_size: int
    base_channels: int
    transformer_layers: int
    transformer_heads: int


PRESET_LARGE = ModelPreset(
    name="large",
    img_size=512,
    base_channels=32,
    transformer_layers=2,
    transformer_heads=4,
)

PRESET_MINI = ModelPreset(
    name="mini",
    img_size=256,
    base_channels=16,
    transformer_layers=1,
    transformer_heads=2,
)

MODEL_PRESETS = {"large": PRESET_LARGE, "mini": PRESET_MINI}

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
