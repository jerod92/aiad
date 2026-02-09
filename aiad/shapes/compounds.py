"""
Compound shape generators: lattices, meshes, concentric shapes, radial
patterns, multi-shape compositions, and shapes using new CAD tools
(mirror, offset, fillet, chamfer, construction lines).

Tool resets (None-tool actions) are inserted between separate line segments
to prevent spurious connecting lines during replay.
"""

import numpy as np

from aiad.config import TOOL_MAP
from aiad.raster import (
    rasterize_line, rasterize_circle, rasterize_rectangle,
    rasterize_regular_polygon, rasterize_ellipse,
    rasterize_fillet_arc, rasterize_chamfer,
    rasterize_construction_line, rasterize_offset_line,
    regular_polygon_vertices, gaussian_blur, mirror_layer,
    rasterize_filled_circle, rasterize_filled_rectangle,
    rasterize_filled_polygon, erase_rectangle,
)
from aiad.shapes._types import ActionStep, ShapeSample


def _rand_pt(margin, S):
    return np.random.randint(margin, S - margin, 2)


def _reset_action():
    """Return a None-tool action to reset active tool state."""
    return ActionStep(TOOL_MAP["None"], 0, 0, 0.0, 0.0, 0.0)


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

    # Horizontal lines (each as a separate 2-click line with reset between)
    for r in range(rows + 1):
        y = int(y0 + r * (y1 - y0) / rows)
        rasterize_line(layer, (x0, y), (x1, y), 1.0, 2)
        if actions:
            actions.append(_reset_action())
        actions.append(ActionStep(TOOL_MAP["Line"], x0, y, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], x1, y, 1.0, 0.0, 0.0))

    # Vertical lines
    for c in range(cols + 1):
        x = int(x0 + c * (x1 - x0) / cols)
        rasterize_line(layer, (x, y0), (x, y1), 1.0, 2)
        actions.append(_reset_action())
        actions.append(ActionStep(TOOL_MAP["Line"], x, y0, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], x, y1, 1.0, 0.0, 0.0))

    # Mark last action as end
    if actions:
        a = actions[-1]
        actions[-1] = ActionStep(a.tool, a.x, a.y, a.click, a.snap, 1.0)
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

    # Horizontal edges
    for r in range(rows + 1):
        for c in range(cols):
            p1, p2 = pts[(r, c)], pts[(r, c + 1)]
            rasterize_line(layer, p1, p2, 1.0, 2)
            if actions:
                actions.append(_reset_action())
            actions.append(ActionStep(TOOL_MAP["Line"], p1[0], p1[1], 1.0, 0.0, 0.0))
            actions.append(ActionStep(TOOL_MAP["Line"], p2[0], p2[1], 1.0, 0.0, 0.0))

    # Vertical edges
    for r in range(rows):
        for c in range(cols + 1):
            p1 = pts[(r, c)]
            p2 = pts[(r + 1, c)]
            rasterize_line(layer, p1, p2, 1.0, 2)
            actions.append(_reset_action())
            actions.append(ActionStep(TOOL_MAP["Line"], p1[0], p1[1], 1.0, 0.0, 0.0))
            actions.append(ActionStep(TOOL_MAP["Line"], p2[0], p2[1], 1.0, 0.0, 0.0))

    # Diagonal edges
    for r in range(rows):
        for c in range(cols):
            if r % 2 == 0:
                p1, p2 = pts[(r, c)], pts[(r + 1, c + 1)]
            else:
                p1, p2 = pts[(r, c + 1)], pts[(r + 1, c)]
            rasterize_line(layer, p1, p2, 1.0, 2)
            actions.append(_reset_action())
            actions.append(ActionStep(TOOL_MAP["Line"], p1[0], p1[1], 1.0, 0.0, 0.0))
            actions.append(ActionStep(TOOL_MAP["Line"], p2[0], p2[1], 1.0, 0.0, 0.0))

    if actions:
        a = actions[-1]
        actions[-1] = ActionStep(a.tool, a.x, a.y, a.click, a.snap, 1.0)
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

    first = True
    for row in range(ny):
        for col in range(nx):
            cx = m + col * dx + (dx / 2 if row % 2 else 0)
            cy = m + row * dy
            if cx + hex_r > S - m or cy + hex_r > S - m:
                continue
            rasterize_regular_polygon(layer, (cx, cy), hex_r, 6,
                                       angle_offset=np.pi / 6, color=1.0, thickness=2)
            verts = regular_polygon_vertices((cx, cy), hex_r, 6, np.pi / 6)
            if not first:
                actions.append(_reset_action())
            first = False
            actions.append(ActionStep(TOOL_MAP["RegPolygon"],
                                      int(cx), int(cy), 1.0, 0.0, 0.0, sides=6))
            actions.append(ActionStep(TOOL_MAP["RegPolygon"],
                                      int(round(verts[0][0])), int(round(verts[0][1])),
                                      1.0, 0.0, 0.0, sides=6))

    if actions:
        a = actions[-1]
        actions[-1] = ActionStep(a.tool, a.x, a.y, a.click, a.snap, 1.0, sides=a.sides)
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
                                  int(center[0]), int(center[1]),
                                  1.0, 0.0, 0.0, sides=n_sides))
        is_last = i == n_rings - 1
        actions.append(ActionStep(TOOL_MAP["RegPolygon"],
                                  int(round(verts[0][0])), int(round(verts[0][1])),
                                  1.0, 0.0, 1.0 if is_last else 0.0,
                                  sides=n_sides))

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
        if actions:
            actions.append(_reset_action())
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
        if actions:
            actions.append(_reset_action())
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
        a = actions[-1]
        actions[-1] = ActionStep(a.tool, a.x, a.y, a.click, a.snap, 1.0)
    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "multi_shape")


# ---------------------------------------------------------------------------
# Symmetric shape (uses Mirror tool)
# ---------------------------------------------------------------------------

def gen_mirror_shape(S: int) -> ShapeSample:
    """Draw one half of a symmetric shape, then mirror across a vertical axis."""
    m = 40
    cx = S // 2
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Draw 2-4 line segments on the left half
    n_segs = np.random.randint(2, 5)
    pts = []
    for _ in range(n_segs + 1):
        px = np.random.randint(m, cx - 5)
        py = np.random.randint(m, S - m)
        pts.append((px, py))

    for i in range(n_segs):
        rasterize_line(layer, pts[i], pts[i + 1], 1.0, 2)
        if i > 0:
            actions.append(_reset_action())
        actions.append(ActionStep(TOOL_MAP["Line"], int(pts[i][0]), int(pts[i][1]),
                                  1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], int(pts[i + 1][0]), int(pts[i + 1][1]),
                                  1.0, 0.0, 0.0))

    # Mirror across vertical center line
    layer = mirror_layer(layer, (cx, 0), (cx, S - 1))
    actions.append(_reset_action())
    actions.append(ActionStep(TOOL_MAP["Mirror"], cx, m, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Mirror"], cx, S - m, 1.0, 0.0, 1.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "mirror_shape")


# ---------------------------------------------------------------------------
# Offset parallel lines
# ---------------------------------------------------------------------------

def gen_offset_lines(S: int) -> ShapeSample:
    """Draw a line and its parallel offset(s)."""
    m = 40
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    p1 = _rand_pt(m, S)
    p2 = _rand_pt(m, S)
    while np.linalg.norm(p2 - p1) < 80:
        p2 = _rand_pt(m, S)

    n_offsets = np.random.randint(1, 4)
    d = p2 - p1
    length = np.linalg.norm(d)
    n_vec = np.array([-d[1], d[0]]) / length

    # Draw original + offsets
    rasterize_line(layer, p1, p2, 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Line"], int(p1[0]), int(p1[1]), 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Line"], int(p2[0]), int(p2[1]), 1.0, 0.0, 0.0))

    for i in range(n_offsets):
        dist = (i + 1) * np.random.randint(20, 50)
        side = np.random.choice([-1, 1])
        op = p1 + n_vec * dist * side
        o1 = p1 + n_vec * dist * side
        o2 = p2 + n_vec * dist * side
        rasterize_line(layer, o1, o2, 1.0, 2)
        actions.append(_reset_action())
        actions.append(ActionStep(TOOL_MAP["Offset"], int(p1[0]), int(p1[1]),
                                  1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Offset"], int(p2[0]), int(p2[1]),
                                  1.0, 0.0, 0.0))
        is_last = i == n_offsets - 1
        actions.append(ActionStep(TOOL_MAP["Offset"], int(round(op[0])), int(round(op[1])),
                                  1.0, 0.0, 1.0 if is_last else 0.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "offset_lines")


# ---------------------------------------------------------------------------
# Rounded rectangle (uses Fillet tool)
# ---------------------------------------------------------------------------

def gen_filleted_rect(S: int) -> ShapeSample:
    """Draw a rectangle with filleted (rounded) corners."""
    m = 40
    c1 = _rand_pt(m + 20, S)
    c2 = _rand_pt(m + 20, S)
    x1, y1 = min(c1[0], c2[0]), min(c1[1], c2[1])
    x2, y2 = max(c1[0], c2[0]), max(c1[1], c2[1])
    if x2 - x1 < 60:
        x2 = min(x1 + np.random.randint(60, 120), S - m)
    if y2 - y1 < 60:
        y2 = min(y1 + np.random.randint(60, 120), S - m)

    rad_hi = max(9, min(25, (x2 - x1) // 3, (y2 - y1) // 3))
    radius = np.random.randint(8, rad_hi)

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Draw the rectangle first
    rasterize_rectangle(layer, (x1, y1), (x2, y2), 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Rectangle"], x1, y1, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Rectangle"], x2, y2, 1.0, 0.0, 0.0))

    # Apply fillet to each corner
    corners = [
        ((x1, y1), (x2, y1), (x1, y2)),  # top-left
        ((x2, y1), (x1, y1), (x2, y2)),  # top-right
        ((x2, y2), (x2, y1), (x1, y2)),  # bottom-right
        ((x1, y2), (x2, y2), (x1, y1)),  # bottom-left
    ]
    for i, (corner, pt_a, pt_b) in enumerate(corners):
        rasterize_fillet_arc(layer, corner, pt_a, pt_b, radius, 1.0, 2)
        actions.append(_reset_action())
        is_last = i == len(corners) - 1
        actions.append(ActionStep(TOOL_MAP["Fillet"], int(corner[0]), int(corner[1]),
                                  1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Fillet"], int(pt_a[0]), int(pt_a[1]),
                                  1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Fillet"], int(pt_b[0]), int(pt_b[1]),
                                  1.0, 0.0, 1.0 if is_last else 0.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "filleted_rect", {"radius": radius})


# ---------------------------------------------------------------------------
# Chamfered rectangle
# ---------------------------------------------------------------------------

def gen_chamfered_rect(S: int) -> ShapeSample:
    """Draw a rectangle with chamfered (beveled) corners."""
    m = 40
    c1 = _rand_pt(m + 20, S)
    c2 = _rand_pt(m + 20, S)
    x1, y1 = min(c1[0], c2[0]), min(c1[1], c2[1])
    x2, y2 = max(c1[0], c2[0]), max(c1[1], c2[1])
    if x2 - x1 < 60:
        x2 = min(x1 + np.random.randint(60, 120), S - m)
    if y2 - y1 < 60:
        y2 = min(y1 + np.random.randint(60, 120), S - m)

    cham_hi = max(9, min(20, (x2 - x1) // 4, (y2 - y1) // 4))
    cham = np.random.randint(8, cham_hi)

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Draw the rectangle
    rasterize_rectangle(layer, (x1, y1), (x2, y2), 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Rectangle"], x1, y1, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Rectangle"], x2, y2, 1.0, 0.0, 0.0))

    # Chamfer each corner
    chamfer_pairs = [
        ((x1 + cham, y1), (x1, y1 + cham)),  # top-left
        ((x2 - cham, y1), (x2, y1 + cham)),  # top-right
        ((x2 - cham, y2), (x2, y2 - cham)),  # bottom-right
        ((x1 + cham, y2), (x1, y2 - cham)),  # bottom-left
    ]
    for i, (pa, pb) in enumerate(chamfer_pairs):
        rasterize_chamfer(layer, pa, pb, 1.0, 2)
        actions.append(_reset_action())
        is_last = i == len(chamfer_pairs) - 1
        actions.append(ActionStep(TOOL_MAP["Chamfer"], int(pa[0]), int(pa[1]),
                                  1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Chamfer"], int(pb[0]), int(pb[1]),
                                  1.0, 0.0, 1.0 if is_last else 0.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "chamfered_rect", {"chamfer": cham})


# ---------------------------------------------------------------------------
# Construction line guide (uses ConstrLine tool)
# ---------------------------------------------------------------------------

def gen_construction_guide(S: int) -> ShapeSample:
    """Draw construction lines with shapes aligned to them."""
    m = 30
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Draw 1-2 construction lines
    n_constr = np.random.randint(1, 3)
    for i in range(n_constr):
        p1 = _rand_pt(m, S)
        p2 = _rand_pt(m, S)
        while np.linalg.norm(p2 - p1) < 80:
            p2 = _rand_pt(m, S)
        rasterize_construction_line(layer, p1, p2, 0.5, 1)
        actions.append(ActionStep(TOOL_MAP["ConstrLine"], int(p1[0]), int(p1[1]),
                                  1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["ConstrLine"], int(p2[0]), int(p2[1]),
                                  1.0, 0.0, 0.0))

    # Draw a shape aligned to the construction
    shape_kind = np.random.choice(["circle", "rectangle"])
    actions.append(_reset_action())
    if shape_kind == "circle":
        c = _rand_pt(m + 30, S)
        r = np.random.randint(25, 70)
        rasterize_circle(layer, c, r, 1.0, 2)
        top = (int(c[0]), int(c[1] - r))
        actions.append(ActionStep(TOOL_MAP["Circle"], int(c[0]), int(c[1]),
                                  1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Circle"], top[0], top[1],
                                  1.0, 0.0, 1.0))
    else:
        c1 = _rand_pt(m, S)
        c2 = _rand_pt(m, S)
        rasterize_rectangle(layer, c1, c2, 1.0, 2)
        actions.append(ActionStep(TOOL_MAP["Rectangle"], int(c1[0]), int(c1[1]),
                                  1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Rectangle"], int(c2[0]), int(c2[1]),
                                  1.0, 0.0, 1.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "construction_guide")


# ---------------------------------------------------------------------------
# Double-walled shape (uses Offset)
# ---------------------------------------------------------------------------

def gen_double_wall(S: int) -> ShapeSample:
    """Draw a rectangle with offset inner walls (like a frame)."""
    m = 40
    x1, y1 = np.random.randint(m, S // 4), np.random.randint(m, S // 4)
    x2, y2 = np.random.randint(S * 3 // 4, S - m), np.random.randint(S * 3 // 4, S - m)
    wall_d = np.random.randint(15, 35)

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Outer rectangle
    rasterize_rectangle(layer, (x1, y1), (x2, y2), 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Rectangle"], x1, y1, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Rectangle"], x2, y2, 1.0, 0.0, 0.0))

    # Inner rectangle (offset)
    ix1, iy1 = x1 + wall_d, y1 + wall_d
    ix2, iy2 = x2 - wall_d, y2 - wall_d
    if ix2 > ix1 + 20 and iy2 > iy1 + 20:
        rasterize_rectangle(layer, (ix1, iy1), (ix2, iy2), 1.0, 2)
        actions.append(_reset_action())
        actions.append(ActionStep(TOOL_MAP["Rectangle"], ix1, iy1, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Rectangle"], ix2, iy2, 1.0, 0.0, 1.0))

    if actions:
        a = actions[-1]
        actions[-1] = ActionStep(a.tool, a.x, a.y, a.click, a.snap, 1.0)
    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "double_wall")


# ---------------------------------------------------------------------------
# Filled circle with outline
# ---------------------------------------------------------------------------

def gen_filled_circle(S: int) -> ShapeSample:
    """Draw a filled circle (solid disc)."""
    m = 50
    center = np.clip(_rand_pt(S // 4, S), S // 4, S * 3 // 4)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    r = np.random.randint(30, max(31, max_r + 1))

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Filled circle (drawn as regular circle — the fill is a visual property)
    rasterize_filled_circle(layer, center, r, 1.0)
    # Outline
    rasterize_circle(layer, center, r, 1.0, 2)

    # Actions: circle center + radius point
    top = (int(center[0]), int(center[1] - r))
    actions.append(ActionStep(TOOL_MAP["Circle"], int(center[0]), int(center[1]),
                              1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Circle"], top[0], top[1], 1.0, 0.0, 1.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "filled_circle")


# ---------------------------------------------------------------------------
# Filled rectangle
# ---------------------------------------------------------------------------

def gen_filled_rectangle(S: int) -> ShapeSample:
    """Draw a filled rectangle."""
    m = 40
    c1 = _rand_pt(m, S)
    c2 = _rand_pt(m, S)
    x1, y1 = min(c1[0], c2[0]), min(c1[1], c2[1])
    x2, y2 = max(c1[0], c2[0]), max(c1[1], c2[1])
    if x2 - x1 < 50:
        x2 = min(x1 + np.random.randint(50, 120), S - m)
    if y2 - y1 < 50:
        y2 = min(y1 + np.random.randint(50, 120), S - m)

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    rasterize_filled_rectangle(layer, (x1, y1), (x2, y2), 1.0)
    rasterize_rectangle(layer, (x1, y1), (x2, y2), 1.0, 2)

    actions.append(ActionStep(TOOL_MAP["Rectangle"], x1, y1, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Rectangle"], x2, y2, 1.0, 0.0, 1.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "filled_rectangle")


# ---------------------------------------------------------------------------
# Filled shapes with trim/subtract (bow-and-arrow style)
# ---------------------------------------------------------------------------

def gen_overlapping_shapes_trimmed(S: int) -> ShapeSample:
    """Draw two overlapping shapes, then trim/erase the overlap region.

    This teaches the model the subtract pattern: draw everything, then
    erase a region so one shape appears to pass *behind* another.
    """
    m = 40
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Draw shape A (the "foreground" — e.g. the bow)
    ax1 = np.random.randint(m, S // 2 - 20)
    ay1 = np.random.randint(m, S // 2 - 20)
    ax2 = np.random.randint(S // 2 + 20, S - m)
    ay2 = np.random.randint(S // 2 + 20, S - m)
    rasterize_filled_rectangle(layer, (ax1, ay1), (ax2, ay2), 1.0)
    rasterize_rectangle(layer, (ax1, ay1), (ax2, ay2), 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Rectangle"], ax1, ay1, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Rectangle"], ax2, ay2, 1.0, 0.0, 0.0))

    # Draw shape B (the "background" — e.g. the arrow)
    kind = np.random.choice(["line", "circle"])
    actions.append(_reset_action())
    if kind == "line":
        # A horizontal line crossing through the rectangle
        lx1 = np.random.randint(m, ax1 - 5) if ax1 > m + 10 else m
        lx2 = min(np.random.randint(ax2 + 5, S - m), S - m)
        ly = np.random.randint(ay1 + 10, ay2 - 10)
        rasterize_line(layer, (lx1, ly), (lx2, ly), 1.0, 2)
        actions.append(ActionStep(TOOL_MAP["Line"], lx1, ly, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], lx2, ly, 1.0, 0.0, 0.0))
    else:
        # A circle overlapping the rectangle
        cx = np.random.randint(ax1 - 30, ax2 + 30)
        cy = np.random.randint(ay1 - 30, ay2 + 30)
        cr = np.random.randint(30, 70)
        rasterize_circle(layer, (cx, cy), cr, 1.0, 2)
        top = (cx, cy - cr)
        actions.append(ActionStep(TOOL_MAP["Circle"], cx, cy, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Circle"], top[0], top[1], 1.0, 0.0, 0.0))

    # Trim: erase the part of shape B that overlaps shape A
    # This makes shape B appear to go behind shape A
    trim_x1 = max(ax1 + 2, 0)
    trim_y1 = max(ay1 + 2, 0)
    trim_x2 = min(ax2 - 2, S)
    trim_y2 = min(ay2 - 2, S)
    erase_rectangle(layer, (trim_x1, trim_y1), (trim_x2, trim_y2))
    # Redraw the foreground rectangle on top
    rasterize_filled_rectangle(layer, (ax1, ay1), (ax2, ay2), 1.0)
    rasterize_rectangle(layer, (ax1, ay1), (ax2, ay2), 1.0, 2)

    actions.append(_reset_action())
    actions.append(ActionStep(TOOL_MAP["Trim"], trim_x1, trim_y1, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Trim"], trim_x2, trim_y2, 1.0, 0.0, 0.0))

    # Redraw shape A outline using Rectangle tool
    actions.append(_reset_action())
    actions.append(ActionStep(TOOL_MAP["Rectangle"], ax1, ay1, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Rectangle"], ax2, ay2, 1.0, 0.0, 1.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "overlapping_trimmed")


# ---------------------------------------------------------------------------
# Cross behind circle (subtract pattern)
# ---------------------------------------------------------------------------

def gen_cross_behind_circle(S: int) -> ShapeSample:
    """Draw crossing lines behind a filled circle — the circle occludes them."""
    m = 50
    center = np.clip(_rand_pt(S // 4, S), S // 4, S * 3 // 4)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    r = np.random.randint(30, max(31, min(80, max_r)))

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Draw crossing lines first
    n_lines = np.random.randint(1, 4)
    for i in range(n_lines):
        p1 = _rand_pt(m, S)
        p2 = _rand_pt(m, S)
        while np.linalg.norm(p2 - p1) < 80:
            p2 = _rand_pt(m, S)
        rasterize_line(layer, p1, p2, 1.0, 2)
        if actions:
            actions.append(_reset_action())
        actions.append(ActionStep(TOOL_MAP["Line"], int(p1[0]), int(p1[1]),
                                  1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], int(p2[0]), int(p2[1]),
                                  1.0, 0.0, 0.0))

    # Erase the circle area (trim the lines where the circle will be)
    trim_x1 = int(center[0] - r)
    trim_y1 = int(center[1] - r)
    trim_x2 = int(center[0] + r)
    trim_y2 = int(center[1] + r)
    erase_rectangle(layer, (trim_x1, trim_y1), (trim_x2, trim_y2))

    actions.append(_reset_action())
    actions.append(ActionStep(TOOL_MAP["Trim"], trim_x1, trim_y1, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Trim"], trim_x2, trim_y2, 1.0, 0.0, 0.0))

    # Draw the filled circle on top
    rasterize_filled_circle(layer, center, r, 1.0)
    rasterize_circle(layer, center, r, 1.0, 2)

    actions.append(_reset_action())
    top = (int(center[0]), int(center[1] - r))
    actions.append(ActionStep(TOOL_MAP["Circle"], int(center[0]), int(center[1]),
                              1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Circle"], top[0], top[1], 1.0, 0.0, 1.0))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "cross_behind_circle")


# ---------------------------------------------------------------------------
# Filled polygon (triangle, pentagon, etc.)
# ---------------------------------------------------------------------------

def gen_filled_polygon(S: int) -> ShapeSample:
    """Draw a filled regular polygon."""
    m = 50
    center = np.clip(_rand_pt(S // 4, S), S // 4, S * 3 // 4)
    max_r = min(center[0] - m, S - center[0] - m, center[1] - m, S - center[1] - m)
    r = np.random.randint(30, max(31, max_r))
    n_sides = np.random.randint(3, 9)
    angle_off = np.random.uniform(0, 2 * np.pi)

    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    verts = regular_polygon_vertices(center, r, n_sides, angle_off)
    rasterize_filled_polygon(layer, verts, 1.0)
    rasterize_regular_polygon(layer, center, r, n_sides, angle_off, 1.0, 2)

    actions.append(ActionStep(TOOL_MAP["RegPolygon"],
                              int(center[0]), int(center[1]),
                              1.0, 0.0, 0.0, sides=n_sides))
    actions.append(ActionStep(TOOL_MAP["RegPolygon"],
                              int(round(verts[0][0])), int(round(verts[0][1])),
                              1.0, 0.0, 1.0, sides=n_sides))

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "filled_polygon")


COMPOUND_GENERATORS = [
    gen_lattice,
    gen_triangle_mesh,
    gen_hex_mesh,
    gen_concentric_circles,
    gen_concentric_polygons,
    gen_radial_pattern,
    gen_multi_shape,
    gen_mirror_shape,
    gen_offset_lines,
    gen_filleted_rect,
    gen_chamfered_rect,
    gen_construction_guide,
    gen_double_wall,
    gen_filled_circle,
    gen_filled_rectangle,
    gen_overlapping_shapes_trimmed,
    gen_cross_behind_circle,
    gen_filled_polygon,
]
