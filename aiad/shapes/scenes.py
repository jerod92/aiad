"""
Complex scene generators: building floor plans, park layouts,
mechanical parts, and architectural sketches.

Each scene is composed of multiple primitives arranged to resemble
real-world 2D CAD drawings.
"""

import numpy as np

from aiad.config import TOOL_MAP
from aiad.raster import (
    rasterize_line, rasterize_circle, rasterize_rectangle,
    rasterize_ellipse, rasterize_arc, rasterize_regular_polygon,
    rasterize_bezier, gaussian_blur,
)
from aiad.shapes._types import ActionStep, ShapeSample


def _rand_range(lo, hi):
    return np.random.randint(lo, max(lo + 1, hi))


# ---------------------------------------------------------------------------
# Building floor plan
# ---------------------------------------------------------------------------

def gen_floorplan(S: int) -> ShapeSample:
    """Generate a simple building floor plan with rooms, doors, and windows."""
    m = 20
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Outer walls
    ox, oy = _rand_range(m, S // 6), _rand_range(m, S // 6)
    ow = _rand_range(S * 2 // 3, S - 2 * m)
    oh = _rand_range(S * 2 // 3, S - 2 * m)
    ox2, oy2 = min(ox + ow, S - m), min(oy + oh, S - m)

    rasterize_rectangle(layer, (ox, oy), (ox2, oy2), 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Rectangle"], ox, oy, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Rectangle"], ox2, oy2, 1.0, 0.0, 0.0))

    # Internal walls (1-3 horizontal, 1-3 vertical)
    n_horiz = np.random.randint(1, 4)
    n_vert = np.random.randint(1, 4)

    for i in range(n_horiz):
        wy = oy + (i + 1) * (oy2 - oy) // (n_horiz + 1)
        # Leave a door gap
        gap_x = np.random.randint(ox + 20, ox2 - 40)
        gap_w = np.random.randint(25, 45)
        rasterize_line(layer, (ox, wy), (gap_x, wy), 1.0, 2)
        rasterize_line(layer, (gap_x + gap_w, wy), (ox2, wy), 1.0, 2)
        actions.append(ActionStep(TOOL_MAP["Line"], ox, wy, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], gap_x, wy, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], gap_x + gap_w, wy, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], ox2, wy, 1.0, 0.0, 0.0))

        # Door arc symbol
        door_cx = gap_x + gap_w // 2
        rasterize_arc(layer, (gap_x, wy), (door_cx, wy - gap_w // 3),
                       (gap_x + gap_w, wy), 0.6, 1)

    for i in range(n_vert):
        wx = ox + (i + 1) * (ox2 - ox) // (n_vert + 1)
        gap_y = np.random.randint(oy + 20, oy2 - 40)
        gap_h = np.random.randint(25, 45)
        rasterize_line(layer, (wx, oy), (wx, gap_y), 1.0, 2)
        rasterize_line(layer, (wx, gap_y + gap_h), (wx, oy2), 1.0, 2)
        actions.append(ActionStep(TOOL_MAP["Line"], wx, oy, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], wx, gap_y, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], wx, gap_y + gap_h, 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], wx, oy2, 1.0, 0.0, 0.0))

    # Windows on outer walls (small double lines)
    n_windows = np.random.randint(2, 6)
    for _ in range(n_windows):
        wall = np.random.choice(["top", "bottom", "left", "right"])
        if wall == "top":
            wx = np.random.randint(ox + 20, ox2 - 30)
            rasterize_line(layer, (wx, oy - 2), (wx + 20, oy - 2), 0.7, 2)
            rasterize_line(layer, (wx, oy + 2), (wx + 20, oy + 2), 0.7, 2)
        elif wall == "bottom":
            wx = np.random.randint(ox + 20, ox2 - 30)
            rasterize_line(layer, (wx, oy2 - 2), (wx + 20, oy2 - 2), 0.7, 2)
            rasterize_line(layer, (wx, oy2 + 2), (wx + 20, oy2 + 2), 0.7, 2)
        elif wall == "left":
            wy = np.random.randint(oy + 20, oy2 - 30)
            rasterize_line(layer, (ox - 2, wy), (ox - 2, wy + 20), 0.7, 2)
            rasterize_line(layer, (ox + 2, wy), (ox + 2, wy + 20), 0.7, 2)
        else:
            wy = np.random.randint(oy + 20, oy2 - 30)
            rasterize_line(layer, (ox2 - 2, wy), (ox2 - 2, wy + 20), 0.7, 2)
            rasterize_line(layer, (ox2 + 2, wy), (ox2 + 2, wy + 20), 0.7, 2)

    if actions:
        actions[-1] = ActionStep(actions[-1].tool, actions[-1].x, actions[-1].y,
                                 1.0, 0.0, 1.0)
    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "floorplan")


# ---------------------------------------------------------------------------
# Park layout
# ---------------------------------------------------------------------------

def gen_park(S: int) -> ShapeSample:
    """Generate a park design with paths, gardens, trees, and benches."""
    m = 20
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Park boundary (rounded rectangle via 4 arcs + 4 lines)
    bx, by = _rand_range(m, S // 6), _rand_range(m, S // 6)
    bw = _rand_range(S * 2 // 3, S - 2 * m)
    bh = _rand_range(S * 2 // 3, S - 2 * m)
    bx2, by2 = min(bx + bw, S - m), min(by + bh, S - m)
    rasterize_rectangle(layer, (bx, by), (bx2, by2), 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Rectangle"], bx, by, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Rectangle"], bx2, by2, 1.0, 0.0, 0.0))

    cx, cy = (bx + bx2) // 2, (by + by2) // 2

    # Central garden (circle or ellipse)
    garden_r = np.random.randint(30, min(80, (bx2 - bx) // 4))
    rasterize_circle(layer, (cx, cy), garden_r, 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Circle"], cx, cy, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Circle"], cx, cy - garden_r, 1.0, 0.0, 0.0))

    # Paths (lines from edges to center area)
    for _ in range(np.random.randint(2, 5)):
        edge = np.random.choice(["top", "bottom", "left", "right"])
        if edge == "top":
            start = (np.random.randint(bx + 20, bx2 - 20), by)
        elif edge == "bottom":
            start = (np.random.randint(bx + 20, bx2 - 20), by2)
        elif edge == "left":
            start = (bx, np.random.randint(by + 20, by2 - 20))
        else:
            start = (bx2, np.random.randint(by + 20, by2 - 20))

        # Path goes toward center with slight curve
        mid_x = (start[0] + cx) // 2 + np.random.randint(-20, 20)
        mid_y = (start[1] + cy) // 2 + np.random.randint(-20, 20)
        rasterize_line(layer, start, (mid_x, mid_y), 0.8, 2)
        rasterize_line(layer, (mid_x, mid_y), (cx, cy), 0.8, 2)
        actions.append(ActionStep(TOOL_MAP["Line"], start[0], start[1], 1.0, 0.0, 0.0))
        actions.append(ActionStep(TOOL_MAP["Line"], cx, cy, 1.0, 0.0, 0.0))

    # Trees (small circles with cross inside)
    n_trees = np.random.randint(3, 8)
    for _ in range(n_trees):
        tx = np.random.randint(bx + 20, bx2 - 20)
        ty = np.random.randint(by + 20, by2 - 20)
        # Skip if too close to center garden
        if np.sqrt((tx - cx) ** 2 + (ty - cy) ** 2) < garden_r + 15:
            continue
        tr = np.random.randint(6, 14)
        rasterize_circle(layer, (tx, ty), tr, 0.9, 1)
        # Cross inside tree
        rasterize_line(layer, (tx - tr // 2, ty), (tx + tr // 2, ty), 0.7, 1)
        rasterize_line(layer, (tx, ty - tr // 2), (tx, ty + tr // 2), 0.7, 1)

    # Benches (small rectangles along paths)
    n_benches = np.random.randint(1, 4)
    for _ in range(n_benches):
        bench_x = np.random.randint(bx + 30, bx2 - 40)
        bench_y = np.random.randint(by + 30, by2 - 40)
        rasterize_rectangle(layer, (bench_x, bench_y),
                           (bench_x + 15, bench_y + 6), 0.8, 1)

    if actions:
        actions[-1] = ActionStep(actions[-1].tool, actions[-1].x, actions[-1].y,
                                 1.0, 0.0, 1.0)
    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "park")


# ---------------------------------------------------------------------------
# Simple mechanical part (bracket / L-shape with holes)
# ---------------------------------------------------------------------------

def gen_mechanical_part(S: int) -> ShapeSample:
    """Generate a simple mechanical bracket with bolt holes."""
    m = 30
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # L-shaped outline
    ox, oy = _rand_range(m, S // 4), _rand_range(m, S // 4)
    w1 = _rand_range(S // 3, S * 2 // 3)
    h1 = _rand_range(S // 4, S // 3)
    w2 = _rand_range(S // 4, S // 3)
    h2 = _rand_range(S // 3, S * 2 // 3)

    # Outer L
    pts = [
        (ox, oy),
        (ox + w1, oy),
        (ox + w1, oy + h1),
        (ox + w2, oy + h1),
        (ox + w2, oy + h2),
        (ox, oy + h2),
    ]
    # Clip to bounds
    pts = [(min(max(p[0], m), S - m), min(max(p[1], m), S - m)) for p in pts]

    for i in range(len(pts)):
        rasterize_line(layer, pts[i], pts[(i + 1) % len(pts)], 1.0, 2)
        actions.append(ActionStep(TOOL_MAP["Line"],
                                  int(pts[i][0]), int(pts[i][1]),
                                  1.0, 1.0 if i == len(pts) - 1 else 0.0,
                                  1.0 if i == len(pts) - 1 else 0.0))

    # Bolt holes (circles)
    hole_r = np.random.randint(5, 12)
    hole_positions = [
        (ox + w2 // 2, oy + h1 // 2),
        (ox + w2 // 2, oy + (h1 + h2) // 2),
        (ox + (w2 + w1) // 2, oy + h1 // 2),
    ]
    for hp in hole_positions:
        hx, hy = int(np.clip(hp[0], m, S - m)), int(np.clip(hp[1], m, S - m))
        rasterize_circle(layer, (hx, hy), hole_r, 0.8, 1)
        # Small cross at center
        rasterize_line(layer, (hx - 3, hy), (hx + 3, hy), 0.5, 1)
        rasterize_line(layer, (hx, hy - 3), (hx, hy + 3), 0.5, 1)

    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "mechanical_part")


# ---------------------------------------------------------------------------
# Garden bed design (circular beds with paths)
# ---------------------------------------------------------------------------

def gen_garden_design(S: int) -> ShapeSample:
    """Generate a garden layout with circular and rectangular beds."""
    m = 25
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Outer rectangle
    bx, by = m + 10, m + 10
    bx2, by2 = S - m - 10, S - m - 10
    rasterize_rectangle(layer, (bx, by), (bx2, by2), 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Rectangle"], bx, by, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Rectangle"], bx2, by2, 1.0, 0.0, 0.0))

    cx, cy = S // 2, S // 2
    # Central circular bed
    cr = np.random.randint(40, S // 6)
    rasterize_circle(layer, (cx, cy), cr, 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Circle"], cx, cy, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Circle"], cx, cy - cr, 1.0, 0.0, 0.0))

    # Inner decorative circle
    rasterize_circle(layer, (cx, cy), cr // 2, 0.7, 1)

    # Corner beds (quarter circles or small rectangles)
    corner_r = np.random.randint(30, 60)
    corners = [(bx, by), (bx2, by), (bx, by2), (bx2, by2)]
    for corner in corners:
        # Small quarter-circle-like shape
        rasterize_circle(layer, corner, corner_r, 0.8, 1)

    # Cross paths
    rasterize_line(layer, (cx, by), (cx, by2), 0.6, 2)
    rasterize_line(layer, (bx, cy), (bx2, cy), 0.6, 2)

    # Diagonal paths
    if np.random.rand() < 0.5:
        rasterize_line(layer, (bx, by), (bx2, by2), 0.4, 1)
        rasterize_line(layer, (bx2, by), (bx, by2), 0.4, 1)

    # Planting dots (small filled circles)
    for _ in range(np.random.randint(5, 15)):
        px = np.random.randint(bx + 10, bx2 - 10)
        py = np.random.randint(by + 10, by2 - 10)
        if np.sqrt((px - cx) ** 2 + (py - cy) ** 2) < cr + 10:
            continue
        rasterize_circle(layer, (px, py), 3, 0.6, -1)  # filled

    if actions:
        actions[-1] = ActionStep(actions[-1].tool, actions[-1].x, actions[-1].y,
                                 1.0, 0.0, 1.0)
    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "garden_design")


# ---------------------------------------------------------------------------
# Simple architectural elevation (building face)
# ---------------------------------------------------------------------------

def gen_building_elevation(S: int) -> ShapeSample:
    """Generate a simple building elevation with walls, windows, and door."""
    m = 20
    layer = np.zeros((S, S), dtype=np.float32)
    actions = []

    # Main wall
    wx, wy = _rand_range(m, S // 6), _rand_range(S // 3, S // 2)
    ww = _rand_range(S * 2 // 3, S - 2 * m)
    wh = _rand_range(S // 3, S // 2)
    wx2, wy2 = min(wx + ww, S - m), min(wy + wh, S - m)
    rasterize_rectangle(layer, (wx, wy), (wx2, wy2), 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Rectangle"], wx, wy, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Rectangle"], wx2, wy2, 1.0, 0.0, 0.0))

    # Roof (triangle)
    roof_peak = ((wx + wx2) // 2, wy - _rand_range(wh // 3, wh // 2))
    rasterize_line(layer, (wx, wy), roof_peak, 1.0, 2)
    rasterize_line(layer, roof_peak, (wx2, wy), 1.0, 2)
    actions.append(ActionStep(TOOL_MAP["Line"], wx, wy, 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Line"], roof_peak[0], roof_peak[1], 1.0, 0.0, 0.0))
    actions.append(ActionStep(TOOL_MAP["Line"], wx2, wy, 1.0, 0.0, 0.0))

    # Door (centered at bottom)
    dcx = (wx + wx2) // 2
    dw, dh = np.random.randint(25, 45), np.random.randint(50, min(80, wh - 10))
    door_y1 = wy2 - dh
    rasterize_rectangle(layer, (dcx - dw // 2, door_y1), (dcx + dw // 2, wy2), 1.0, 2)
    # Door arc
    rasterize_arc(layer, (dcx - dw // 2, door_y1),
                  (dcx, door_y1 - dw // 4), (dcx + dw // 2, door_y1), 0.8, 1)

    # Windows (2-4 small rectangles)
    n_windows = np.random.randint(2, 5)
    win_w, win_h = np.random.randint(20, 35), np.random.randint(25, 40)
    for i in range(n_windows):
        win_x = wx + (i + 1) * (wx2 - wx) // (n_windows + 1) - win_w // 2
        win_y = wy + (wy2 - wy) // 4
        if abs(win_x + win_w // 2 - dcx) < dw:
            continue
        rasterize_rectangle(layer, (win_x, win_y), (win_x + win_w, win_y + win_h), 0.9, 1)
        # Window cross
        rasterize_line(layer, (win_x + win_w // 2, win_y),
                       (win_x + win_w // 2, win_y + win_h), 0.6, 1)
        rasterize_line(layer, (win_x, win_y + win_h // 2),
                       (win_x + win_w, win_y + win_h // 2), 0.6, 1)

    if actions:
        actions[-1] = ActionStep(actions[-1].tool, actions[-1].x, actions[-1].y,
                                 1.0, 0.0, 1.0)
    layer = gaussian_blur(layer)
    return ShapeSample(layer, actions, "building_elevation")


SCENE_GENERATORS = [
    gen_floorplan,
    gen_park,
    gen_mechanical_part,
    gen_garden_design,
    gen_building_elevation,
]
