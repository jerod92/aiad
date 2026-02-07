"""Tests for the CAD drawing environment."""

import numpy as np
import pytest

from aiad.config import TOOL_MAP
from aiad.env import MixedCADEnvironment, thick_line_iou


IMG_SIZE = 256


@pytest.fixture
def env():
    return MixedCADEnvironment(img_size=IMG_SIZE, max_steps=10)


class TestReset:
    def test_returns_tuple(self, env):
        obs_tuple = env.reset()
        assert len(obs_tuple) == 3

    def test_obs_shape(self, env):
        obs, pt, pc = env.reset()
        assert obs.shape == (6, IMG_SIZE, IMG_SIZE)

    def test_initial_tool_is_none(self, env):
        _, pt, pc = env.reset()
        assert pt == TOOL_MAP["None"]
        assert pc == 0.0


class TestStep:
    def test_step_returns_4_tuple(self, env):
        env.reset()
        action = {"x": 100, "y": 100, "tool": TOOL_MAP["Line"],
                  "click": 1.0, "snap": 0.0, "end": 0.0}
        result = env.step(action)
        assert len(result) == 4

    def test_max_steps_terminates(self, env):
        env.reset()
        noop = {"x": 128, "y": 128, "tool": TOOL_MAP["None"],
                "click": 0.0, "snap": 0.0, "end": 0.0}
        done = False
        for _ in range(20):
            _, _, done, _ = env.step(noop)
            if done:
                break
        assert done

    def test_end_action_terminates(self, env):
        env.reset()
        action = {"x": 128, "y": 128, "tool": TOOL_MAP["None"],
                  "click": 0.0, "snap": 0.0, "end": 1.0}
        _, _, done, info = env.step(action)
        assert done
        assert "iou" in info


class TestLineDrawing:
    def test_line_draws_pixels(self, env):
        env.reset()
        env.step({"x": 50, "y": 50, "tool": TOOL_MAP["Line"],
                  "click": 1.0, "snap": 0.0, "end": 0.0})
        env.step({"x": 150, "y": 150, "tool": TOOL_MAP["Line"],
                  "click": 1.0, "snap": 0.0, "end": 0.0})
        assert env.drawing_layer.sum() > 0


class TestCircleDrawing:
    def test_circle_draws_pixels(self, env):
        env.reset()
        env.step({"x": 128, "y": 128, "tool": TOOL_MAP["Circle"],
                  "click": 1.0, "snap": 0.0, "end": 0.0})
        env.step({"x": 170, "y": 128, "tool": TOOL_MAP["Circle"],
                  "click": 1.0, "snap": 0.0, "end": 0.0})
        assert env.drawing_layer.sum() > 0


class TestRectangleDrawing:
    def test_rectangle_draws_pixels(self, env):
        env.reset()
        env.step({"x": 50, "y": 50, "tool": TOOL_MAP["Rectangle"],
                  "click": 1.0, "snap": 0.0, "end": 0.0})
        env.step({"x": 150, "y": 150, "tool": TOOL_MAP["Rectangle"],
                  "click": 1.0, "snap": 0.0, "end": 0.0})
        assert env.drawing_layer.sum() > 0


class TestSplineDrawing:
    def test_spline_draws_pixels(self, env):
        env.reset()
        env.step({"x": 50, "y": 128, "tool": TOOL_MAP["Spline"],
                  "click": 1.0, "snap": 0.0, "end": 0.0})
        env.step({"x": 100, "y": 50, "tool": TOOL_MAP["Spline"],
                  "click": 1.0, "snap": 0.0, "end": 0.0})
        env.step({"x": 150, "y": 200, "tool": TOOL_MAP["Spline"],
                  "click": 1.0, "snap": 0.0, "end": 0.0})
        env.step({"x": 200, "y": 128, "tool": TOOL_MAP["Spline"],
                  "click": 1.0, "snap": 1.0, "end": 0.0})
        assert env.drawing_layer.sum() > 0


class TestThickLineIoU:
    def test_perfect_overlap(self):
        S = 64
        img = np.zeros((S, S), dtype=np.float32)
        from aiad.raster import rasterize_line
        rasterize_line(img, (10, 32), (50, 32), 1.0, 1)
        iou = thick_line_iou(img, img.copy(), thickness=5)
        assert iou > 0.5

    def test_no_drawing_gives_zero(self):
        S = 64
        target = np.zeros((S, S), dtype=np.float32)
        from aiad.raster import rasterize_line
        rasterize_line(target, (10, 32), (50, 32), 1.0, 1)
        empty = np.zeros((S, S), dtype=np.float32)
        iou = thick_line_iou(target, empty, thickness=5)
        assert iou < 0.01

    def test_near_miss_gets_partial_credit(self):
        S = 64
        target = np.zeros((S, S), dtype=np.float32)
        drawing = np.zeros((S, S), dtype=np.float32)
        from aiad.raster import rasterize_line
        rasterize_line(target, (10, 32), (50, 32), 1.0, 1)
        rasterize_line(drawing, (10, 34), (50, 34), 1.0, 1)
        iou = thick_line_iou(target, drawing, thickness=10)
        assert iou > 0.1
