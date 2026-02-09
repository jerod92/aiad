"""
Low-level rasterization utilities backed by OpenCV.

All drawing functions operate on float32 numpy arrays in [0, 1] range.
Anti-aliased rendering (LINE_AA) is used wherever OpenCV supports it to
produce smooth edges that are friendly to convolutional feature extraction.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def rasterize_line(img, p1, p2, color=1.0, thickness=1):
    """Draw an anti-aliased line segment on a float32 numpy image."""
    cv2.line(
        img,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        float(color),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def rasterize_dashed_line(img, p1, p2, color=1.0, thickness=1,
                          dash_len=10, gap_len=6):
    """Draw a dashed line segment on a float32 numpy image."""
    p1 = np.array([float(p1[0]), float(p1[1])])
    p2 = np.array([float(p2[0]), float(p2[1])])
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1:
        return img
    unit = d / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg_len = dash_len if drawing else gap_len
        end_pos = min(pos + seg_len, length)
        if drawing:
            a = p1 + unit * pos
            b = p1 + unit * end_pos
            cv2.line(img,
                     (int(round(a[0])), int(round(a[1]))),
                     (int(round(b[0])), int(round(b[1]))),
                     float(color), thickness, lineType=cv2.LINE_AA)
        pos = end_pos
        drawing = not drawing
    return img


def rasterize_circle(img, center, radius, color=1.0, thickness=1):
    """Draw an anti-aliased circle on a float32 numpy image."""
    cv2.circle(
        img,
        (int(round(center[0])), int(round(center[1]))),
        int(max(round(radius), 1)),
        float(color),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def rasterize_rectangle(img, corner1, corner2, color=1.0, thickness=1):
    """Draw an axis-aligned rectangle (two opposite corners)."""
    x1, y1 = int(round(corner1[0])), int(round(corner1[1]))
    x2, y2 = int(round(corner2[0])), int(round(corner2[1]))
    cv2.rectangle(img, (x1, y1), (x2, y2), float(color), thickness, lineType=cv2.LINE_AA)
    return img


def rasterize_ellipse(img, center, axes, angle=0.0, color=1.0, thickness=1):
    """Draw an anti-aliased ellipse.

    Parameters
    ----------
    center : (x, y)
    axes   : (semi_major, semi_minor)
    angle  : rotation in degrees
    """
    cv2.ellipse(
        img,
        (int(round(center[0])), int(round(center[1]))),
        (int(max(round(axes[0]), 1)), int(max(round(axes[1]), 1))),
        float(angle), 0, 360, float(color), thickness, lineType=cv2.LINE_AA,
    )
    return img


def rasterize_arc(img, p_start, p_mid, p_end, color=1.0, thickness=1):
    """Draw a three-point arc (start, mid-point on arc, end).

    Falls back to a polyline approximation using the circumscribed circle
    through the three points.
    """
    pts = _three_point_arc_samples(p_start, p_mid, p_end, num_samples=64)
    if pts is None:
        # Degenerate (collinear) — just draw a line
        rasterize_line(img, p_start, p_end, color, thickness)
        return img
    for i in range(len(pts) - 1):
        rasterize_line(img, pts[i], pts[i + 1], color, thickness)
    return img


def rasterize_regular_polygon(img, center, radius, num_sides, angle_offset=0.0,
                               color=1.0, thickness=1):
    """Draw a regular polygon inscribed in a circle of given radius."""
    pts = _regular_polygon_vertices(center, radius, num_sides, angle_offset)
    for i in range(num_sides):
        rasterize_line(img, pts[i], pts[(i + 1) % num_sides], color, thickness)
    return img


def rasterize_star(img, center, outer_r, inner_r, num_points, angle_offset=0.0,
                   color=1.0, thickness=1):
    """Draw a star with alternating outer/inner vertices."""
    pts = _star_vertices(center, outer_r, inner_r, num_points, angle_offset)
    n = len(pts)
    for i in range(n):
        rasterize_line(img, pts[i], pts[(i + 1) % n], color, thickness)
    return img


def rasterize_bezier(img, control_points, color=1.0, thickness=1, num_samples=80):
    """Draw a cubic (or quadratic) Bezier curve through its control points."""
    pts = _bezier_samples(control_points, num_samples)
    for i in range(len(pts) - 1):
        rasterize_line(img, pts[i], pts[i + 1], color, thickness)
    return img


def rasterize_polyline(img, points, color=1.0, thickness=1, closed=False):
    """Draw connected line segments through a list of points."""
    n = len(points)
    segs = n if closed else n - 1
    for i in range(segs):
        rasterize_line(img, points[i], points[(i + 1) % n], color, thickness)
    return img


def rasterize_fillet_arc(img, corner, pt_a, pt_b, radius, color=1.0, thickness=1):
    """Draw a fillet (rounded corner) arc between two lines meeting at *corner*.

    The arc replaces the corner with a smooth circular arc of given *radius*
    tangent to both edges.
    """
    cx, cy = float(corner[0]), float(corner[1])
    ax, ay = float(pt_a[0]), float(pt_a[1])
    bx, by = float(pt_b[0]), float(pt_b[1])
    va = np.array([ax - cx, ay - cy], dtype=np.float64)
    vb = np.array([bx - cx, by - cy], dtype=np.float64)
    la = np.linalg.norm(va)
    lb = np.linalg.norm(vb)
    if la < 1 or lb < 1:
        return img
    va /= la
    vb /= lb
    bisect = va + vb
    bn = np.linalg.norm(bisect)
    if bn < 1e-6:
        return img
    bisect /= bn
    half_angle = np.arccos(np.clip(np.dot(va, vb), -1, 1)) / 2
    if abs(np.sin(half_angle)) < 1e-6:
        return img
    d = radius / np.sin(half_angle)
    arc_center = np.array([cx, cy]) + bisect * d
    ta = np.array([cx, cy]) + va * (radius / np.tan(half_angle))
    tb = np.array([cx, cy]) + vb * (radius / np.tan(half_angle))
    a_start = np.arctan2(ta[1] - arc_center[1], ta[0] - arc_center[0])
    a_end = np.arctan2(tb[1] - arc_center[1], tb[0] - arc_center[0])
    if a_end < a_start:
        a_end += 2 * np.pi
    if a_end - a_start > np.pi:
        a_start, a_end = a_end, a_start + 2 * np.pi
    angles = np.linspace(a_start, a_end, 32)
    pts = [(arc_center[0] + radius * np.cos(a), arc_center[1] + radius * np.sin(a))
           for a in angles]
    for i in range(len(pts) - 1):
        rasterize_line(img, pts[i], pts[i + 1], color, thickness)
    return img


def rasterize_chamfer(img, pt1, pt2, color=1.0, thickness=1):
    """Draw a chamfer (beveled corner) as a straight line between two points."""
    rasterize_line(img, pt1, pt2, color, thickness)
    return img


def rasterize_construction_line(img, p1, p2, color=0.5, thickness=1,
                                 dash_len=8, gap_len=5):
    """Draw a construction line that extends to canvas edges (dashed)."""
    S = img.shape[0]
    p1 = np.array([float(p1[0]), float(p1[1])])
    p2 = np.array([float(p2[0]), float(p2[1])])
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1:
        return img
    unit = d / length
    # Extend in both directions to edges
    max_extent = S * 2
    ext_a = p1 - unit * max_extent
    ext_b = p1 + unit * max_extent
    rasterize_dashed_line(img, ext_a, ext_b, color, thickness, dash_len, gap_len)
    return img


def mirror_layer(img, axis_start, axis_end):
    """Mirror the drawing layer across the axis defined by two points.

    Returns a new image with the original + mirrored content.
    """
    S = img.shape[0]
    ax1 = np.array([float(axis_start[0]), float(axis_start[1])])
    ax2 = np.array([float(axis_end[0]), float(axis_end[1])])
    d = ax2 - ax1
    length = np.linalg.norm(d)
    if length < 1:
        return img
    n = np.array([-d[1], d[0]]) / length  # normal to axis

    result = img.copy()
    ys, xs = np.where(img > 0.05)
    for px, py in zip(xs, ys):
        v = np.array([px - ax1[0], py - ax1[1]])
        dist = np.dot(v, n)
        mx = int(round(px - 2 * dist * n[0]))
        my = int(round(py - 2 * dist * n[1]))
        if 0 <= mx < S and 0 <= my < S:
            result[my, mx] = max(result[my, mx], img[py, px])
    return result


def rasterize_offset_line(img, p1, p2, offset_pt, color=1.0, thickness=1):
    """Draw a line parallel to (p1->p2) at perpendicular distance to offset_pt."""
    p1 = np.array([float(p1[0]), float(p1[1])])
    p2 = np.array([float(p2[0]), float(p2[1])])
    op = np.array([float(offset_pt[0]), float(offset_pt[1])])
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1:
        return img
    n_vec = np.array([-d[1], d[0]]) / length
    dist = np.dot(op - p1, n_vec)
    # Draw original line
    rasterize_line(img, p1, p2, color, thickness)
    # Draw offset line
    o1 = p1 + n_vec * dist
    o2 = p2 + n_vec * dist
    rasterize_line(img, o1, o2, color, thickness)
    return img


def erase_rectangle(img, corner1, corner2):
    """Erase (zero out) a rectangular region — the subtract/trim operation.

    Used for removing geometry that should appear *behind* other geometry,
    e.g. an arrow behind a bow.
    """
    x1, y1 = int(round(min(corner1[0], corner2[0]))), int(round(min(corner1[1], corner2[1])))
    x2, y2 = int(round(max(corner1[0], corner2[0]))), int(round(max(corner1[1], corner2[1])))
    S = img.shape[0]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(S, x2), min(S, y2)
    img[y1:y2, x1:x2] = 0.0
    return img


def rasterize_filled_circle(img, center, radius, color=1.0):
    """Draw a filled circle (thickness=-1)."""
    cv2.circle(
        img,
        (int(round(center[0])), int(round(center[1]))),
        int(max(round(radius), 1)),
        float(color),
        -1,
        lineType=cv2.LINE_AA,
    )
    return img


def rasterize_filled_rectangle(img, corner1, corner2, color=1.0):
    """Draw a filled rectangle (thickness=-1)."""
    x1, y1 = int(round(corner1[0])), int(round(corner1[1]))
    x2, y2 = int(round(corner2[0])), int(round(corner2[1]))
    cv2.rectangle(img, (x1, y1), (x2, y2), float(color), -1, lineType=cv2.LINE_AA)
    return img


def rasterize_filled_polygon(img, vertices, color=1.0):
    """Draw a filled polygon from a list of (x, y) vertices."""
    pts = np.array([(int(round(v[0])), int(round(v[1]))) for v in vertices], dtype=np.int32)
    cv2.fillPoly(img, [pts], float(color), lineType=cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Cursor & post-processing
# ---------------------------------------------------------------------------

def draw_cursor_np(img, cx, cy, size=8, color=1.0, thickness=1):
    """Draw a '+' crosshair cursor.  Kept small and clean for conv-friendliness."""
    cx, cy = int(round(cx)), int(round(cy))
    cv2.line(img, (cx, cy - size), (cx, cy + size), float(color), thickness, cv2.LINE_AA)
    cv2.line(img, (cx - size, cy), (cx + size, cy), float(color), thickness, cv2.LINE_AA)
    return img


def gaussian_blur(img, ksize=5):
    """Apply Gaussian blur to soften rasterized lines for smooth gradients."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


# ---------------------------------------------------------------------------
# Geometry helpers (public — also used by shape generators)
# ---------------------------------------------------------------------------

def regular_polygon_vertices(center, radius, num_sides, angle_offset=0.0):
    """Return vertices of a regular polygon as a list of (x, y) tuples."""
    return _regular_polygon_vertices(center, radius, num_sides, angle_offset)


def star_vertices(center, outer_r, inner_r, num_points, angle_offset=0.0):
    """Return alternating outer/inner vertices for a star."""
    return _star_vertices(center, outer_r, inner_r, num_points, angle_offset)


def bezier_samples(control_points, num_samples=80):
    """Evaluate a Bezier curve at uniform t values. Public wrapper."""
    return _bezier_samples(control_points, num_samples)


# ---------------------------------------------------------------------------
# Geometry helpers (internal)
# ---------------------------------------------------------------------------

def _regular_polygon_vertices(center, radius, num_sides, angle_offset=0.0):
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False) + angle_offset
    return [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]


def _star_vertices(center, outer_r, inner_r, num_points, angle_offset=0.0):
    cx, cy = center
    verts = []
    for i in range(num_points):
        a_out = angle_offset + 2 * np.pi * i / num_points
        a_in = a_out + np.pi / num_points
        verts.append((cx + outer_r * np.cos(a_out), cy + outer_r * np.sin(a_out)))
        verts.append((cx + inner_r * np.cos(a_in), cy + inner_r * np.sin(a_in)))
    return verts


def _bezier_samples(control_points, num_samples=80):
    """Evaluate a Bezier curve (any degree) via De Casteljau's algorithm."""
    cp = np.array(control_points, dtype=np.float64)
    n = len(cp) - 1
    ts = np.linspace(0, 1, num_samples)
    pts = []
    for t in ts:
        tmp = cp.copy()
        for k in range(n):
            tmp[:n - k] = (1 - t) * tmp[:n - k] + t * tmp[1:n - k + 1]
        pts.append(tuple(tmp[0]))
    return pts


def _circumscribed_circle(p1, p2, p3):
    """Return (center_x, center_y, radius) of circle through three points, or None."""
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    cx, cy = float(p3[0]), float(p3[1])
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-8:
        return None
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) +
          (cx ** 2 + cy ** 2) * (ay - by)) / D
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) +
          (cx ** 2 + cy ** 2) * (bx - ax)) / D
    r = np.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)
    return ux, uy, r


def _three_point_arc_samples(p_start, p_mid, p_end, num_samples=64):
    """Sample points along an arc defined by three points."""
    result = _circumscribed_circle(p_start, p_mid, p_end)
    if result is None:
        return None
    cx, cy, r = result
    a_start = np.arctan2(p_start[1] - cy, p_start[0] - cx)
    a_mid = np.arctan2(p_mid[1] - cy, p_mid[0] - cx)
    a_end = np.arctan2(p_end[1] - cy, p_end[0] - cx)

    def _normalise(a):
        return a % (2 * np.pi)

    a_start_n = _normalise(a_start)
    a_mid_n = _normalise(a_mid)
    a_end_n = _normalise(a_end)

    def _between_ccw(a, lo, hi):
        if lo <= hi:
            return lo <= a <= hi
        return a >= lo or a <= hi

    if _between_ccw(a_mid_n, a_start_n, a_end_n):
        if a_end_n < a_start_n:
            a_end_n += 2 * np.pi
    else:
        if a_end_n > a_start_n:
            a_end_n -= 2 * np.pi

    angles = np.linspace(a_start_n, a_end_n, num_samples)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
