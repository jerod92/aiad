"""Shared dataclasses for the shapes package (avoids circular imports)."""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ActionStep:
    """One atomic CAD action in a drawing sequence."""
    tool: int          # tool index (from config.TOOL_MAP)
    x: int             # cursor x
    y: int             # cursor y
    click: float       # 1.0 = click, 0.0 = hover
    snap: float        # 1.0 = snap / close shape
    end: float         # 1.0 = end drawing session
    sides: int = 0     # for RegPolygon: number of sides (0 = default/unused)


@dataclass
class ShapeSample:
    """Complete description of one synthetic training target."""
    target_layer: np.ndarray             # float32 [H, W], rasterised target
    actions: List[ActionStep]            # ordered CAD action sequence
    category: str = ""                   # human-readable shape name
    metadata: dict = field(default_factory=dict)
