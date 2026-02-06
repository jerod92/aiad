"""Tests for rasterization utilities."""

import numpy as np
import pytest

from aiad.raster import rasterize_line, rasterize_circle, draw_cursor_np, gaussian_blur


@pytest.fixture
def blank():
    return np.zeros((64, 64), dtype=np.float32)


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


class TestDrawCursor:
    def test_draws_crosshair(self, blank):
        draw_cursor_np(blank, 32, 32, size=5)
        assert blank[32, 32] > 0  # center
        assert blank[27, 32] > 0  # above
        assert blank[32, 27] > 0  # left


class TestGaussianBlur:
    def test_spreads_values(self, blank):
        blank[32, 32] = 1.0
        blurred = gaussian_blur(blank)
        assert blurred[32, 32] < 1.0
        assert blurred[31, 32] > 0
