"""
Bezier / B-spline curve generators.

The key challenge: the model sees the *evaluated* curve on the target image
but must learn to predict the *control points*, which generally do NOT lie
on the curve (for cubic Bezier the two inner handles are off-curve).
"""

import numpy as np

from aiad.config import TOOL_MAP
from aiad.raster import rasterize_bezier, rasterize_line, gaussian_blur
from aiad.shapes._types import ActionStep, ShapeSample


def _rand_pt(margin, S):
    return np.random.randint(margin, S - margin, 2).astype(float)


# ---------------------------------------------------------------------------
# Quadratic Bezier (3 control points)
# ---------------------------------------------------------------------------

def gen_quadratic_bezier(S: int) -> ShapeSample:
    m = 40
    # P0, P1 (off-curve handle), P2
    p0 = _rand_pt(m, S)
    p2 = _rand_pt(m, S)
    while np.linalg.norm(p2 - p0) < 80:
        p2 = _rand_pt(m, S)
    # Handle: offset from midpoint to create curvature
    mid = (p0 + p2) / 2
    offset = np.random.uniform(-0.8, 0.8, 2) * np.linalg.norm(p2 - p0) * 0.5
    p1 = np.clip(mid + offset, m, S - m)

    cps = [p0, p1, p2]
    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_bezier(layer, cps, 1.0, 2)
    layer = gaussian_blur(layer)

    actions = [
        ActionStep(TOOL_MAP["Spline"], int(p0[0]), int(p0[1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Spline"], int(p1[0]), int(p1[1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Spline"], int(p2[0]), int(p2[1]), 1.0, 0.0, 1.0),
    ]
    return ShapeSample(layer, actions, "quad_bezier", {"control_points": cps})


# ---------------------------------------------------------------------------
# Cubic Bezier (4 control points)
# ---------------------------------------------------------------------------

def gen_cubic_bezier(S: int) -> ShapeSample:
    m = 40
    p0 = _rand_pt(m, S)
    p3 = _rand_pt(m, S)
    while np.linalg.norm(p3 - p0) < 80:
        p3 = _rand_pt(m, S)

    # Two off-curve handles
    span = p3 - p0
    perp = np.array([-span[1], span[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-8)

    t1, t2 = np.random.uniform(0.2, 0.45), np.random.uniform(0.55, 0.8)
    h1 = np.random.uniform(0.2, 0.7) * np.linalg.norm(span) * np.random.choice([-1, 1])
    h2 = np.random.uniform(0.2, 0.7) * np.linalg.norm(span) * np.random.choice([-1, 1])
    p1 = np.clip(p0 + t1 * span + h1 * perp, m, S - m)
    p2 = np.clip(p0 + t2 * span + h2 * perp, m, S - m)

    cps = [p0, p1, p2, p3]
    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_bezier(layer, cps, 1.0, 2)
    layer = gaussian_blur(layer)

    actions = [
        ActionStep(TOOL_MAP["Spline"], int(p0[0]), int(p0[1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Spline"], int(p1[0]), int(p1[1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Spline"], int(p2[0]), int(p2[1]), 1.0, 0.0, 0.0),
        ActionStep(TOOL_MAP["Spline"], int(p3[0]), int(p3[1]), 1.0, 0.0, 1.0),
    ]
    return ShapeSample(layer, actions, "cubic_bezier", {"control_points": cps})


# ---------------------------------------------------------------------------
# S-curve (two cubic segments chained)
# ---------------------------------------------------------------------------

def gen_s_curve(S: int) -> ShapeSample:
    m = 40
    p0 = _rand_pt(m, S)
    p6 = _rand_pt(m, S)
    while np.linalg.norm(p6 - p0) < 120:
        p6 = _rand_pt(m, S)

    mid = (p0 + p6) / 2
    span = p6 - p0
    perp = np.array([-span[1], span[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-8)

    d = np.linalg.norm(span) * 0.3
    p1 = np.clip(p0 + 0.33 * span + d * perp, m, S - m)
    p2 = np.clip(mid + d * perp * 0.5, m, S - m)
    p3 = mid.copy()
    p4 = np.clip(mid - d * perp * 0.5, m, S - m)
    p5 = np.clip(p0 + 0.67 * span - d * perp, m, S - m)

    # Draw as two cubic Bezier segments
    cps1 = [p0, p1, p2, p3]
    cps2 = [p3, p4, p5, p6]
    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_bezier(layer, cps1, 1.0, 2)
    rasterize_bezier(layer, cps2, 1.0, 2)
    layer = gaussian_blur(layer)

    all_cps = [p0, p1, p2, p3, p4, p5, p6]
    actions = []
    for i, cp in enumerate(all_cps):
        is_last = i == len(all_cps) - 1
        actions.append(ActionStep(
            TOOL_MAP["Spline"], int(cp[0]), int(cp[1]),
            1.0, 0.0, 1.0 if is_last else 0.0,
        ))
    return ShapeSample(layer, actions, "s_curve", {"control_points": all_cps})


# ---------------------------------------------------------------------------
# Closed Bezier loop
# ---------------------------------------------------------------------------

def gen_closed_bezier(S: int) -> ShapeSample:
    m = 50
    center = np.array([S / 2, S / 2])
    # Generate 4 anchor points around a noisy circle
    n_anchors = 4
    angles = np.linspace(0, 2 * np.pi, n_anchors, endpoint=False)
    angles += np.random.uniform(-0.3, 0.3, n_anchors)
    base_r = np.random.uniform(S * 0.15, S * 0.35)

    anchors = []
    for a in angles:
        r = base_r + np.random.uniform(-base_r * 0.3, base_r * 0.3)
        px = np.clip(center[0] + r * np.cos(a), m, S - m)
        py = np.clip(center[1] + r * np.sin(a), m, S - m)
        anchors.append(np.array([px, py]))

    # Build cubic segments between consecutive anchors with handles
    layer = np.zeros((S, S), dtype=np.float32)
    all_cps = []
    for i in range(n_anchors):
        a0 = anchors[i]
        a1 = anchors[(i + 1) % n_anchors]
        mid = (a0 + a1) / 2
        span = a1 - a0
        perp = np.array([-span[1], span[0]]) * 0.3
        h1 = np.clip(a0 + 0.33 * span + perp * np.random.uniform(-1, 1), m, S - m)
        h2 = np.clip(a0 + 0.67 * span + perp * np.random.uniform(-1, 1), m, S - m)
        segment = [a0, h1, h2, a1]
        rasterize_bezier(layer, segment, 1.0, 2)
        if i == 0:
            all_cps.extend([a0, h1, h2])
        else:
            all_cps.extend([h1, h2])
    layer = gaussian_blur(layer)

    actions = []
    for i, cp in enumerate(all_cps):
        is_last = i == len(all_cps) - 1
        actions.append(ActionStep(
            TOOL_MAP["Spline"], int(cp[0]), int(cp[1]),
            1.0, 1.0 if is_last else 0.0, 1.0 if is_last else 0.0,
        ))
    return ShapeSample(layer, actions, "closed_bezier")


# ---------------------------------------------------------------------------
# Random wiggly spline (5-8 control points)
# ---------------------------------------------------------------------------

def gen_random_spline(S: int) -> ShapeSample:
    m = 40
    n_cp = np.random.randint(5, 9)
    # Generate control points along a rough path
    start = _rand_pt(m, S)
    end = _rand_pt(m, S)
    while np.linalg.norm(end - start) < 100:
        end = _rand_pt(m, S)

    cps = [start]
    for i in range(1, n_cp - 1):
        t = i / (n_cp - 1)
        base = start * (1 - t) + end * t
        jitter = np.random.uniform(-S * 0.2, S * 0.2, 2)
        cp = np.clip(base + jitter, m, S - m)
        cps.append(cp)
    cps.append(end)

    layer = np.zeros((S, S), dtype=np.float32)
    rasterize_bezier(layer, cps, 1.0, 2)
    layer = gaussian_blur(layer)

    actions = []
    for i, cp in enumerate(cps):
        is_last = i == len(cps) - 1
        actions.append(ActionStep(
            TOOL_MAP["Spline"], int(cp[0]), int(cp[1]),
            1.0, 0.0, 1.0 if is_last else 0.0,
        ))
    return ShapeSample(layer, actions, "random_spline", {"n_control_points": n_cp})


SPLINE_GENERATORS = [
    gen_quadratic_bezier,
    gen_cubic_bezier,
    gen_s_curve,
    gen_closed_bezier,
    gen_random_spline,
]
