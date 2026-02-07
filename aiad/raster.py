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
