"""Tests for rasterization utilities (original + new primitives)."""

import numpy as np
import pytest

from aiad.raster import (
    rasterize_line, rasterize_circle, rasterize_rectangle,
    rasterize_ellipse, rasterize_arc, rasterize_regular_polygon,
    rasterize_star, rasterize_bezier, rasterize_polyline,
    draw_cursor_np, gaussian_blur,
    regular_polygon_vertices, star_vertices, bezier_samples,
)


@pytest.fixture
def blank():
    return np.zeros((64, 64), dtype=np.float32)


@pytest.fixture
def blank_large():
    return np.zeros((256, 256), dtype=np.float32)


class TestRasterizeLine:
    def test_draws_pixels(self, blank):
        rasterize_line(blank, (10, 10), (50, 50))
        assert blank.sum() > 0

    def test_horizontal(self, blank):
        rasterize_line(blank, (0, 32), (63, 32))
        assert blank[32, :].sum() > 0

    def test_thickness(self, blank):
        thin = blank.copy()
        rasterize_line(thin, (10, 10), (50, 50), thickness=1)
        thick = blank.copy()
        rasterize_line(thick, (10, 10), (50, 50), thickness=5)
        assert thick.sum() > thin.sum()

    def test_out_of_bounds_no_crash(self, blank):
        rasterize_line(blank, (-10, -10), (100, 100))


class TestRasterizeCircle:
    def test_draws_pixels(self, blank):
        rasterize_circle(blank, (32, 32), 15)
        assert blank.sum() > 0

    def test_small_radius(self, blank):
        rasterize_circle(blank, (32, 32), 0.5)

    def test_thickness(self, blank):
        thin = blank.copy()
        rasterize_circle(thin, (32, 32), 20, thickness=1)
        thick = blank.copy()
        rasterize_circle(thick, (32, 32), 20, thickness=5)
        assert thick.sum() > thin.sum()


class TestRasterizeRectangle:
    def test_draws_pixels(self, blank):
        rasterize_rectangle(blank, (10, 10), (50, 50))
        assert blank.sum() > 0

    def test_different_corners(self, blank):
        rasterize_rectangle(blank, (5, 5), (60, 30), 1.0, 2)
        assert blank.sum() > 0


class TestRasterizeEllipse:
    def test_draws_pixels(self, blank):
        rasterize_ellipse(blank, (32, 32), (20, 10))
        assert blank.sum() > 0

    def test_rotated(self, blank):
        rasterize_ellipse(blank, (32, 32), (20, 10), angle=45.0)
        assert blank.sum() > 0


class TestRasterizeArc:
    def test_draws_pixels(self, blank_large):
        rasterize_arc(blank_large, (50, 128), (128, 50), (200, 128))
        assert blank_large.sum() > 0

    def test_collinear_fallback(self, blank):
        rasterize_arc(blank, (10, 32), (32, 32), (50, 32))
        assert blank.sum() > 0


class TestRasterizeRegularPolygon:
    def test_triangle(self, blank_large):
        rasterize_regular_polygon(blank_large, (128, 128), 60, 3)
        assert blank_large.sum() > 0

    def test_hexagon(self, blank_large):
        rasterize_regular_polygon(blank_large, (128, 128), 60, 6)
        assert blank_large.sum() > 0


class TestRasterizeStar:
    def test_5_point(self, blank_large):
        rasterize_star(blank_large, (128, 128), 80, 40, 5)
        assert blank_large.sum() > 0


class TestRasterizeBezier:
    def test_quadratic(self, blank_large):
        rasterize_bezier(blank_large, [(50, 200), (128, 30), (200, 200)])
        assert blank_large.sum() > 0

    def test_cubic(self, blank_large):
        rasterize_bezier(blank_large, [(30, 200), (80, 30), (180, 30), (220, 200)])
        assert blank_large.sum() > 0


class TestRasterizePolyline:
    def test_open(self, blank):
        rasterize_polyline(blank, [(10, 10), (30, 50), (50, 10)])
        assert blank.sum() > 0

    def test_closed(self, blank):
        open_img = blank.copy()
        rasterize_polyline(open_img, [(10, 10), (30, 50), (50, 10)], closed=False)
        closed_img = blank.copy()
        rasterize_polyline(closed_img, [(10, 10), (30, 50), (50, 10)], closed=True)
        assert closed_img.sum() > open_img.sum()


class TestDrawCursor:
    def test_draws_crosshair(self, blank):
        draw_cursor_np(blank, 32, 32, size=5)
        assert blank[32, 32] > 0


class TestGaussianBlur:
    def test_spreads_values(self, blank):
        blank[32, 32] = 1.0
        blurred = gaussian_blur(blank)
        assert blurred[32, 32] < 1.0
        assert blurred[31, 32] > 0


class TestGeometryHelpers:
    def test_polygon_vertices_count(self):
        verts = regular_polygon_vertices((100, 100), 50, 6)
        assert len(verts) == 6

    def test_star_vertices_count(self):
        verts = star_vertices((100, 100), 80, 40, 5)
        assert len(verts) == 10

    def test_bezier_samples_count(self):
        pts = bezier_samples([(0, 0), (50, 100), (100, 0)], num_samples=20)
        assert len(pts) == 20

    def test_bezier_endpoints(self):
        pts = bezier_samples([(10, 20), (50, 80), (90, 20)], num_samples=50)
        assert abs(pts[0][0] - 10) < 1
        assert abs(pts[-1][0] - 90) < 1
