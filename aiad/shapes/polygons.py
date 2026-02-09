"""
Regular polygons and star generators.

These use the RegPolygon tool (index 6) for regular n-gons and the Line tool
for stars (since stars have non-uniform vertex distances).

ActionStep.sides carries the number of sides for replay consistency.
"""

import numpy as np

from aiad.config import TOOL_MAP
from aiad.raster import (
    rasterize_regular_polygon, rasterize_star,
    regular_polygon_vertices, star_vertices,
    rasterize_line, gaussian_blur,
)
from aiad.shapes._types import ActionStep, ShapeSample


def _rand_center(margin, S):
    return np.random.randint(margin, S - margin, 2)


# ---------------------------------------------------------------------------
# Regular polygon (3-12 sides)
# ---------------------------------------------------------------------------

def gen_regular_polygon(S: int) -> ShapeSample:
    m = 50
    center = _rand_center(m + 40, S)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    radius = np.random.randint(40, max(41, max_r + 1))
    n_sides = np.random.randint(3, 13)
    angle_off = np.random.uniform(0, 2 * np.pi)

    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_regular_polygon(layer, center, radius, n_sides, angle_off, 1.0, 2)
    layer = gaussian_blur(layer)

    verts = regular_polygon_vertices(center, radius, n_sides, angle_off)
    first_v = verts[0]
    actions = [
        ActionStep(TOOL_MAP["RegPolygon"], int(center[0]), int(center[1]),
                   1.0, 0.0, 0.0, sides=n_sides),
        ActionStep(TOOL_MAP["RegPolygon"], int(round(first_v[0])), int(round(first_v[1])),
                   1.0, 0.0, 1.0, sides=n_sides),
    ]
    return ShapeSample(layer, actions, f"reg_polygon_{n_sides}",
                       {"n_sides": n_sides, "radius": radius})


# ---------------------------------------------------------------------------
# Pentagon specifically
# ---------------------------------------------------------------------------

def gen_pentagon(S: int) -> ShapeSample:
    m = 50
    center = _rand_center(m + 40, S)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    radius = np.random.randint(50, max(51, max_r + 1))
    angle_off = np.random.uniform(0, 2 * np.pi)

    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_regular_polygon(layer, center, radius, 5, angle_off, 1.0, 2)
    layer = gaussian_blur(layer)

    verts = regular_polygon_vertices(center, radius, 5, angle_off)
    actions = [
        ActionStep(TOOL_MAP["RegPolygon"], int(center[0]), int(center[1]),
                   1.0, 0.0, 0.0, sides=5),
        ActionStep(TOOL_MAP["RegPolygon"], int(round(verts[0][0])), int(round(verts[0][1])),
                   1.0, 0.0, 1.0, sides=5),
    ]
    return ShapeSample(layer, actions, "pentagon", {"n_sides": 5, "radius": radius})


# ---------------------------------------------------------------------------
# Hexagon specifically
# ---------------------------------------------------------------------------

def gen_hexagon(S: int) -> ShapeSample:
    m = 50
    center = _rand_center(m + 40, S)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    radius = np.random.randint(50, max(51, max_r + 1))
    angle_off = np.random.uniform(0, 2 * np.pi)

    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_regular_polygon(layer, center, radius, 6, angle_off, 1.0, 2)
    layer = gaussian_blur(layer)

    verts = regular_polygon_vertices(center, radius, 6, angle_off)
    actions = [
        ActionStep(TOOL_MAP["RegPolygon"], int(center[0]), int(center[1]),
                   1.0, 0.0, 0.0, sides=6),
        ActionStep(TOOL_MAP["RegPolygon"], int(round(verts[0][0])), int(round(verts[0][1])),
                   1.0, 0.0, 1.0, sides=6),
    ]
    return ShapeSample(layer, actions, "hexagon", {"n_sides": 6, "radius": radius})


# ---------------------------------------------------------------------------
# Stars (5, 6, 8 points)
# ---------------------------------------------------------------------------

def _gen_star(S: int, n_points: int) -> ShapeSample:
    m = 50
    center = _rand_center(m + 50, S)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    outer_r = np.random.randint(50, max(51, max_r + 1))
    inner_r = np.random.randint(max(15, outer_r // 4), max(16, outer_r * 2 // 3 + 1))
    angle_off = np.random.uniform(0, 2 * np.pi)

    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_star(layer, center, outer_r, inner_r, n_points, angle_off, 1.0, 2)
    layer = gaussian_blur(layer)

    # Stars are drawn as closed polylines using the Line tool
    verts = star_vertices(center, outer_r, inner_r, n_points, angle_off)
    actions = []
    for i, v in enumerate(verts):
        is_last = i == len(verts) - 1
        actions.append(ActionStep(
            TOOL_MAP["Line"], int(round(v[0])), int(round(v[1])),
            1.0, 1.0 if is_last else 0.0, 1.0 if is_last else 0.0,
        ))
    return ShapeSample(layer, actions, f"star_{n_points}")


def gen_star_5(S: int) -> ShapeSample:
    return _gen_star(S, 5)


def gen_star_6(S: int) -> ShapeSample:
    return _gen_star(S, 6)


def gen_star_8(S: int) -> ShapeSample:
    return _gen_star(S, 8)


POLYGON_GENERATORS = [
    gen_regular_polygon,
    gen_pentagon,
    gen_hexagon,
    gen_star_5,
    gen_star_6,
    gen_star_8,
]
