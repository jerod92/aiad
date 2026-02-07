"""
Shape generation package for AIAD.

Each sub-module exposes a list of generator functions.  Every generator
returns a ``ShapeSample`` — a dataclass that carries the rasterised target,
the ordered list of CAD actions needed to reproduce it, and metadata.

Usage::

    from aiad.shapes import ALL_GENERATORS, random_shape
    sample = random_shape(img_size=512)
"""

import numpy as np

from aiad.shapes._types import ActionStep, ShapeSample  # noqa: F401
from aiad.shapes.primitives import PRIMITIVE_GENERATORS
from aiad.shapes.polygons import POLYGON_GENERATORS
from aiad.shapes.splines import SPLINE_GENERATORS
from aiad.shapes.compounds import COMPOUND_GENERATORS
from aiad.shapes.scenes import SCENE_GENERATORS


ALL_GENERATORS = (
    PRIMITIVE_GENERATORS
    + POLYGON_GENERATORS
    + SPLINE_GENERATORS
    + COMPOUND_GENERATORS
    + SCENE_GENERATORS
)


def random_shape(img_size: int = 512) -> ShapeSample:
    """Pick a random generator and produce a sample."""
    gen = ALL_GENERATORS[np.random.randint(len(ALL_GENERATORS))]
    return gen(img_size)
