"""
Global configuration: tool registry, model presets, image sizes.

The tool list is intentionally oversized (24 slots) so future tools can be
added without retraining from scratch.  Only indices 0-15, 19 are *active*
today (excluding 8-10); the rest are reserved placeholders whose logits the
model still learns to suppress during training.
"""

import torch
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Tool registry (24 slots — 12 active, 12 reserved)
# ---------------------------------------------------------------------------

TOOLS = [
    # --- Core drawing tools (0-7) ---
    "None",         # 0  idle / tool deselect / reset active state
    "Line",         # 1  polyline drawing (click anchors, snap to close)
    "Circle",       # 2  center + radius-point
    "Rectangle",    # 3  two-corner axis-aligned rectangle
    "Arc",          # 4  three-point arc (start, mid, end)
    "Ellipse",      # 5  center + semi-major + semi-minor
    "RegPolygon",   # 6  regular polygon (center + vertex, num_sides encoded)
    "Spline",       # 7  cubic Bezier (place control points, snap to finish)
    # --- Modification tools (8-15) ---
    "Move",         # 8  (reserved)
    "Rotate",       # 9  (reserved)
    "Scale",        # 10 (reserved)
    "Mirror",       # 11 axis_start + axis_end → mirror drawing
    "Offset",       # 12 line_start + line_end + offset_pt → parallel line
    "Trim",         # 13 subtract / erase: two corners define erase rectangle
    "Fillet",       # 14 corner_pt + radius_pt → rounded corner arc
    "Chamfer",      # 15 pt1 + pt2 → beveled corner line
    # --- Annotation / utility tools (16-23) ---
    "Hatch",        # 16 (reserved)
    "Dimension",    # 17 (reserved)
    "Array",        # 18 (reserved)
    "ConstrLine",   # 19 two points → infinite construction line (thin/dashed)
    "Text",         # 20 (reserved)
    "Select",       # 21 (reserved)
    "Extend",       # 22 (reserved)
    "Undo",         # 23 (reserved)
]

TOOL_MAP = {name: i for i, name in enumerate(TOOLS)}
NUM_TOOLS = len(TOOLS)                     # 24
ACTIVE_TOOLS = set(range(8)) | {11, 12, 13, 14, 15, 19}  # core + modification + utility

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
