"""
Primitive shape generators: lines, polylines, polygons, circles,
rectangles, arcs, ellipses.
"""

import numpy as np

from aiad.config import TOOL_MAP
from aiad.raster import (
    rasterize_line, rasterize_circle, rasterize_rectangle,
    rasterize_ellipse, rasterize_arc, gaussian_blur,
)
from aiad.shapes._types import ActionStep, ShapeSample


def _rand_pt(margin, S):
    return np.random.randint(margin, S - margin, 2)


def _well_separated(num, margin, S, min_dist=50):
    pts = [_rand_pt(margin, S)]
    for _ in range(num - 1):
        for _try in range(100):
            p = _rand_pt(margin, S)
            if np.linalg.norm(p - pts[-1]) >= min_dist:
                break
        pts.append(p)
    return pts


# ---------------------------------------------------------------------------
# Single line
# ---------------------------------------------------------------------------

def gen_line(S: int) -> ShapeSample:
    m = 30
    pts = _well_separated(2, m, S, 80)
    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_line(layer, pts[0], pts[1], 1.0, 2)
    layer = gaussian_blur(layer)

    actions = [
        ActionStep(TOOL_MAP["Line"], int(pts[0][0]), int(pts[0][1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Line"], int(pts[1][0]), int(pts[1][1]), 1.0, 0.0, 1.0),
    ]
    return ShapeSample(layer, actions, "line")


# ---------------------------------------------------------------------------
# Polyline (open, 2-6 segments)
# ---------------------------------------------------------------------------

def gen_polyline(S: int) -> ShapeSample:
    m = 30
    n_seg = np.random.randint(2, 7)
    pts = _well_separated(n_seg + 1, m, S, 50)
    layer = np.zeros((S, S), dtype=np.float32)
    for i in range(n_seg):
        rasterize_line(layer, pts[i], pts[i + 1], 1.0, 2)
    layer = gaussian_blur(layer)

    actions = []
    for i, p in enumerate(pts):
        is_last = i == len(pts) - 1
        actions.append(ActionStep(
            TOOL_MAP["Line"], int(p[0]), int(p[1]),
            1.0, 0.0, 1.0 if is_last else 0.0,
        ))
    return ShapeSample(layer, actions, "polyline")


# ---------------------------------------------------------------------------
# Closed polygon (3-8 sides)
# ---------------------------------------------------------------------------

def gen_polygon(S: int) -> ShapeSample:
    m = 30
    n_sides = np.random.randint(3, 9)
    pts = _well_separated(n_sides, m, S, 40)
    layer = np.zeros((S, S), dtype=np.float32)
    for i in range(n_sides):
        rasterize_line(layer, pts[i], pts[(i + 1) % n_sides], 1.0, 2)
    layer = gaussian_blur(layer)

    actions = []
    for i, p in enumerate(pts):
        is_last = i == n_sides - 1
        actions.append(ActionStep(
            TOOL_MAP["Line"], int(p[0]), int(p[1]),
            1.0, 1.0 if is_last else 0.0, 1.0 if is_last else 0.0,
        ))
    return ShapeSample(layer, actions, "polygon")


# ---------------------------------------------------------------------------
# Circle
# ---------------------------------------------------------------------------

def gen_circle(S: int) -> ShapeSample:
    m = 30
    safe = m + 50
    center = np.random.randint(safe, S - safe, 2)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    radius = np.random.randint(40, max(41, max_r + 1))
    top = np.array([center[0], center[1] - radius])

    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_circle(layer, center, radius, 1.0, 2)
    layer = gaussian_blur(layer)

    actions = [
        ActionStep(TOOL_MAP["Circle"], int(center[0]), int(center[1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Circle"], int(top[0]), int(top[1]), 1.0, 0.0, 1.0),
    ]
    return ShapeSample(layer, actions, "circle", {"center": center, "radius": radius})


# ---------------------------------------------------------------------------
# Rectangle (axis-aligned)
# ---------------------------------------------------------------------------

def gen_rectangle(S: int) -> ShapeSample:
    m = 30
    pts = _well_separated(2, m, S, 80)
    c1, c2 = pts[0], pts[1]
    # Ensure minimum rectangle size
    if abs(c1[0] - c2[0]) < 40:
        c2[0] = c1[0] + np.random.choice([-1, 1]) * np.random.randint(40, 100)
        c2[0] = np.clip(c2[0], m, S - m)
    if abs(c1[1] - c2[1]) < 40:
        c2[1] = c1[1] + np.random.choice([-1, 1]) * np.random.randint(40, 100)
        c2[1] = np.clip(c2[1], m, S - m)

    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_rectangle(layer, c1, c2, 1.0, 2)
    layer = gaussian_blur(layer)

    actions = [
        ActionStep(TOOL_MAP["Rectangle"], int(c1[0]), int(c1[1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Rectangle"], int(c2[0]), int(c2[1]), 1.0, 0.0, 1.0),
    ]
    return ShapeSample(layer, actions, "rectangle")


# ---------------------------------------------------------------------------
# Three-point arc
# ---------------------------------------------------------------------------

def gen_arc(S: int) -> ShapeSample:
    m = 40
    # Generate three non-collinear points
    for _ in range(50):
        pts = _well_separated(3, m, S, 60)
        v1 = np.array(pts[1]) - np.array(pts[0])
        v2 = np.array(pts[2]) - np.array(pts[0])
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        if cross > 500:
            break

    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_arc(layer, pts[0], pts[1], pts[2], 1.0, 2)
    layer = gaussian_blur(layer)

    actions = [
        ActionStep(TOOL_MAP["Arc"], int(pts[0][0]), int(pts[0][1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Arc"], int(pts[1][0]), int(pts[1][1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Arc"], int(pts[2][0]), int(pts[2][1]), 1.0, 0.0, 1.0),
    ]
    return ShapeSample(layer, actions, "arc")


# ---------------------------------------------------------------------------
# Ellipse
# ---------------------------------------------------------------------------

def gen_ellipse(S: int) -> ShapeSample:
    m = 40
    center = _rand_pt(m + 50, S)
    max_ax = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    a = np.random.randint(30, max(31, max_ax + 1))
    b = np.random.randint(20, max(21, int(a * 0.9) + 1))
    if np.random.rand() < 0.5:
        a, b = b, a
    angle = np.random.uniform(0, 180)

    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_ellipse(layer, center, (a, b), angle, 1.0, 2)
    layer = gaussian_blur(layer)

    # The action sequence: click center, click semi-major endpoint, click semi-minor endpoint
    rad = np.radians(angle)
    ep_a = (center[0] + a * np.cos(rad), center[1] + a * np.sin(rad))
    ep_b = (center[0] - b * np.sin(rad), center[1] + b * np.cos(rad))
    actions = [
        ActionStep(TOOL_MAP["Ellipse"], int(center[0]), int(center[1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Ellipse"], int(round(ep_a[0])), int(round(ep_a[1])), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Ellipse"], int(round(ep_b[0])), int(round(ep_b[1])), 1.0, 0.0, 1.0),
    ]
    return ShapeSample(layer, actions, "ellipse",
                       {"center": center, "axes": (a, b), "angle": angle})


PRIMITIVE_GENERATORS = [
    gen_line,
    gen_polyline,
    gen_polygon,
    gen_circle,
    gen_rectangle,
    gen_arc,
    gen_ellipse,
]
