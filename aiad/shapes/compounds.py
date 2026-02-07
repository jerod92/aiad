"""
Compound shape generators: lattices, meshes, concentric shapes, radial
patterns, and multi-shape compositions.
"""

import numpy as np

from aiad.config import TOOL_MAP
from aiad.raster import (
    rasterize_line, rasterize_circle, rasterize_rectangle,
    rasterize_regular_polygon, rasterize_ellipse,
    regular_polygon_vertices, gaussian_blur,
)
from aiad.shapes._types import ActionStep, ShapeSample


def _rand_pt(margin, S):
    return np.random.randint(margin, S - margin, 2)


# ---------------------------------------------------------------------------
# Rectangular lattice (grid)
# ---------------------------------------------------------------------------

def gen_lattice(S: int) -> ShapeSample:
    m = 30
    rows = np.random.randint(2, 5)
    cols = np.random.randint(2, 5)
    x0, y0 = np.random.randint(m, S // 4), np.random.randint(m, S // 4)
    x1 = np.random.randint(S * 3 // 4, S - m)
    y1 = np.random.randint(S * 3 // 4, S - m)

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Horizontal lines
    for r in range(rows + 1):
        y = int(y0 + r * (y1 - y0) / rows)
        rasterize_line(layer, (x0, y), (x1, y), 1.0, 2)
        actions.append(ActionStep(TOOL_MAP["Line"], x0, y, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], x1, y, 1.0, 0.0, 0.0))
    # Vertical lines
    for c in range(cols + 1):
        x = int(x0 + c * (x1 - x0) / cols)
        rasterize_line(layer, (x, y0), (x, y1), 1.0, 2)
        actions.append(ActionStep(TOOL_MAP["Line"], x, y0, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], x, y1, 1.0, 0.0, 0.0))

    # Mark last action as end
    if actions:
        actions[-1] = ActionStep(actions[-1].tool, actions[-1].x, actions[-1].y,
                                 1.0, 0.0, 1.0)
    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "lattice",
                       {"rows": rows, "cols": cols})


# ---------------------------------------------------------------------------
# Triangular mesh
# ---------------------------------------------------------------------------

def gen_triangle_mesh(S: int) -> ShapeSample:
    m = 30
    rows = np.random.randint(2, 4)
    cols = np.random.randint(2, 5)
    x0, y0 = np.random.randint(m, S // 4), np.random.randint(m, S // 4)
    dx = np.random.randint(60, max(61, (S - 2 * m) // cols))
    dy = np.random.randint(60, max(61, (S - 2 * m) // rows))

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    pts = {}
    for r in range(rows + 1):
        for c in range(cols + 1):
            x = x0 + c * dx + (dy // 2 if r % 2 else 0)
            y = y0 + r * dy
            x = min(x, S - m)
            y = min(y, S - m)
            pts[(r, c)] = (x, y)

    for r in range(rows + 1):
        for c in range(cols):
            p1, p2 = pts[(r, c)], pts[(r, c + 1)]
            rasterize_line(layer, p1, p2, 1.0, 2)
            actions.append(ActionStep(TOOL_MAP["Line"], p1[0], p1[1], 1.0, 0.0, 0.0))
            actions.append(ActionStep(TOOL_MAP["Line"], p2[0], p2[1], 1.0, 0.0, 0.0))

    for r in range(rows):
        for c in range(cols + 1):
            p1 = pts[(r, c)]
            p2 = pts[(r + 1, c)]
            rasterize_line(layer, p1, p2, 1.0, 2)
            actions.append(ActionStep(TOOL_MAP["Line"], p1[0], p1[1], 1.0, 0.0, 0.0))
            actions.append(ActionStep(TOOL_MAP["Line"], p2[0], p2[1], 1.0, 0.0, 0.0))
            # Diagonal
            if c < cols:
                p3 = pts[(r + 1, c + 1)] if r % 2 == 0 else pts[(r, c + 1)]
                rasterize_line(layer, p1, pts[(r + 1, c + 1 if r % 2 == 0 else c)], 1.0, 2)

    if actions:
        actions[-1] = ActionStep(actions[-1].tool, actions[-1].x, actions[-1].y,
                                 1.0, 0.0, 1.0)
    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "triangle_mesh")


# ---------------------------------------------------------------------------
# Hexagonal mesh (honeycomb)
# ---------------------------------------------------------------------------

def gen_hex_mesh(S: int) -> ShapeSample:
    m = 30
    hex_r = np.random.randint(25, 55)
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    dx = hex_r * 1.5
    dy = hex_r * np.sqrt(3)
    nx = int((S - 2 * m) / dx) + 1
    ny = int((S - 2 * m) / dy) + 1

    for row in range(ny):
        for col in range(nx):
            cx = m + col * dx + (dx / 2 if row % 2 else 0)
            cy = m + row * dy
            if cx + hex_r > S - m or cy + hex_r > S - m:
                continue
            rasterize_regular_polygon(layer, (cx, cy), hex_r, 6,
                                       angle_offset=np.pi / 6, color=1.0, thickness=2)
            actions.append(ActionStep(TOOL_MAP["RegPolygon"],
                                      int(cx), int(cy), 1.0, 0.0, 0.0))

    if actions:
        actions[-1] = ActionStep(actions[-1].tool, actions[-1].x, actions[-1].y,
                                 1.0, 0.0, 1.0)
    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "hex_mesh", {"hex_radius": hex_r})


# ---------------------------------------------------------------------------
# Concentric circles
# ---------------------------------------------------------------------------

def gen_concentric_circles(S: int) -> ShapeSample:
    m = 30
    center = _rand_pt(S // 4, S)
    center = np.clip(center, S // 4, S * 3 // 4)
    n_rings = np.random.randint(3, 7)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    radii = sorted(np.random.randint(20, max(21, max_r + 1), n_rings))
    # Ensure minimum gap
    for i in range(1, len(radii)):
        if radii[i] - radii[i - 1] < 15:
            radii[i] = radii[i - 1] + 15

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []
    for i, r in enumerate(radii):
        if r > max_r:
            break
        rasterize_circle(layer, center, r, 1.0, 2)
        top = (int(center[0]), int(center[1] - r))
        actions.append(ActionStep(TOOL_MAP["Circle"], int(center[0]), int(center[1]),
                                  1.0, 0.0, 0.0))
        is_last = i == len(radii) - 1 or (i + 1 < len(radii) and radii[i + 1] > max_r)
        actions.append(ActionStep(TOOL_MAP["Circle"], top[0], top[1],
                                  1.0, 0.0, 1.0 if is_last else 0.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "concentric_circles")


# ---------------------------------------------------------------------------
# Concentric polygons
# ---------------------------------------------------------------------------

def gen_concentric_polygons(S: int) -> ShapeSample:
    m = 40
    center = np.clip(_rand_pt(S // 4, S), S // 4, S * 3 // 4)
    n_sides = np.random.randint(3, 9)
    n_rings = np.random.randint(2, 5)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    angle_off = np.random.uniform(0, 2 * np.pi)

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []
    for i in range(n_rings):
        r = int(max_r * (i + 1) / n_rings)
        rasterize_regular_polygon(layer, center, r, n_sides, angle_off, 1.0, 2)
        verts = regular_polygon_vertices(center, r, n_sides, angle_off)
        actions.append(ActionStep(TOOL_MAP["RegPolygon"],
                                  int(center[0]), int(center[1]), 1.0, 0.0, 0.0))
        is_last = i == n_rings - 1
        actions.append(ActionStep(TOOL_MAP["RegPolygon"],
                                  int(round(verts[0][0])), int(round(verts[0][1])),
                                  1.0, 0.0, 1.0 if is_last else 0.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "concentric_polygons")


# ---------------------------------------------------------------------------
# Radial pattern (spokes from center)
# ---------------------------------------------------------------------------

def gen_radial_pattern(S: int) -> ShapeSample:
    m = 30
    center = np.clip(_rand_pt(S // 4, S), S // 4, S * 3 // 4)
    n_spokes = np.random.randint(4, 13)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    spoke_len = np.random.randint(max_r // 2, max(max_r // 2 + 1, max_r + 1))
    angle_off = np.random.uniform(0, 2 * np.pi / n_spokes)

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []
    for i in range(n_spokes):
        a = angle_off + 2 * np.pi * i / n_spokes
        end = (center[0] + spoke_len * np.cos(a), center[1] + spoke_len * np.sin(a))
        rasterize_line(layer, center, end, 1.0, 2)
        actions.append(ActionStep(TOOL_MAP["Line"], int(center[0]), int(center[1]),
                                  1.0, 0.0, 0.0))
        is_last = i == n_spokes - 1
        actions.append(ActionStep(TOOL_MAP["Line"], int(round(end[0])), int(round(end[1])),
                                  1.0, 0.0, 1.0 if is_last else 0.0))

    # Optionally add outer circle
    if np.random.rand() < 0.5:
        rasterize_circle(layer, center, spoke_len, 1.0, 2)

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "radial_pattern")


# ---------------------------------------------------------------------------
# Multi-shape scatter (2-4 random primitives on one canvas)
# ---------------------------------------------------------------------------

def gen_multi_shape(S: int) -> ShapeSample:
    m = 40
    n_shapes = np.random.randint(2, 5)
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    for _ in range(n_shapes):
        kind = np.random.choice(["circle", "rectangle", "line"])
        if kind == "circle":
            c = _rand_pt(m + 30, S)
            max_r = min(c[0] - m, S - c[0] - m, c[1] - m, S - c[1] - m)
            r = np.random.randint(20, max(21, min(80, max_r + 1)))
            rasterize_circle(layer, c, r, 1.0, 2)
            top = (int(c[0]), int(c[1] - r))
            actions.append(ActionStep(TOOL_MAP["Circle"], int(c[0]), int(c[1]),
                                      1.0, 0.0, 0.0))
            actions.append(ActionStep(TOOL_MAP["Circle"], top[0], top[1],
                                      1.0, 0.0, 0.0))
        elif kind == "rectangle":
            c1 = _rand_pt(m, S)
            c2 = _rand_pt(m, S)
            rasterize_rectangle(layer, c1, c2, 1.0, 2)
            actions.append(ActionStep(TOOL_MAP["Rectangle"], int(c1[0]), int(c1[1]),
                                      1.0, 0.0, 0.0))
            actions.append(ActionStep(TOOL_MAP["Rectangle"], int(c2[0]), int(c2[1]),
                                      1.0, 0.0, 0.0))
        else:
            p1 = _rand_pt(m, S)
            p2 = _rand_pt(m, S)
            rasterize_line(layer, p1, p2, 1.0, 2)
            actions.append(ActionStep(TOOL_MAP["Line"], int(p1[0]), int(p1[1]),
                                      1.0, 0.0, 0.0))
            actions.append(ActionStep(TOOL_MAP["Line"], int(p2[0]), int(p2[1]),
                                      1.0, 0.0, 0.0))

    if actions:
        actions[-1] = ActionStep(actions[-1].tool, actions[-1].x, actions[-1].y,
                                 1.0, 0.0, 1.0)
    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "multi_shape")


COMPOUND_GENERATORS = [
    gen_lattice,
    gen_triangle_mesh,
    gen_hex_mesh,
    gen_concentric_circles,
    gen_concentric_polygons,
    gen_radial_pattern,
    gen_multi_shape,
]
