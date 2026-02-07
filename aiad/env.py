"""
CAD drawing environment for PPO training.

The agent sees a target raster image and must reproduce it using CAD tools.
Supports Line, Circle, Rectangle, Arc, Ellipse, RegPolygon, and Spline.
At episode end, reward = thick-line IoU between the agent's drawing and the
target.
"""

import numpy as np
import cv2

from aiad.config import TOOL_MAP, IMG_SIZE
from aiad.raster import (
    rasterize_line, rasterize_circle, rasterize_rectangle,
    rasterize_ellipse, rasterize_arc, rasterize_bezier,
    rasterize_regular_polygon, draw_cursor_np, gaussian_blur,
)
from aiad.shapes import random_shape


def thick_line_iou(target_layer, drawing_layer, thickness=10):
    """Compute IoU between target and drawing using dilated (thick) masks.

    Both the ground-truth shape and the model's drawing are thickened so that
    near-miss parallel lines still receive partial credit.
    """
    S = target_layer.shape[0]
    kernel = np.ones((thickness, thickness), np.uint8)

    target_mask = cv2.dilate((target_layer > 0.1).astype(np.uint8), kernel, iterations=1)
    drawn_mask = cv2.dilate((drawing_layer > 0.1).astype(np.uint8), kernel, iterations=1)

    intersection = np.logical_and(target_mask > 0, drawn_mask > 0).sum()
    union = np.logical_or(target_mask > 0, drawn_mask > 0).sum()
    return float(intersection / (union + 1e-6))


class MixedCADEnvironment:
    """Gym-like environment for CAD tracing with mixed shape targets.

    Uses the ``aiad.shapes`` package to generate diverse targets.
    """

    def __init__(self, img_size=IMG_SIZE, max_steps=50, reward_thickness=10):
        self.img_size = img_size
        self.max_steps = max_steps
        self.reward_thickness = reward_thickness
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        S = self.img_size
        shape = random_shape(S)
        self.target_layer = shape.target_layer

        self.drawing_layer = np.zeros((S, S), dtype=np.float32)
        self.ghost_layer = np.zeros((S, S), dtype=np.float32)
        self.cursor_pos = (S // 2, S // 2)

        # Tool-specific active state
        self.line_start = None
        self.polyline_start = None
        self.circle_center = None
        self.rect_start = None
        self.arc_pts = []
        self.ellipse_pts = []
        self.regpoly_center = None
        self.spline_pts = []

        self.steps = 0
        self.prev_tool = TOOL_MAP["None"]
        self.prev_click = 0.0

        return self._get_obs()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self):
        S = self.img_size
        target_rgb = np.stack([self.target_layer] * 3, axis=0)
        cursor = np.zeros((S, S), dtype=np.float32)
        cx, cy = self.cursor_pos
        draw_cursor_np(cursor, cx, cy)
        obs = np.concatenate([
            target_rgb,
            self.drawing_layer[np.newaxis],
            self.ghost_layer[np.newaxis],
            cursor[np.newaxis],
        ], axis=0)
        return obs, self.prev_tool, self.prev_click

    # ------------------------------------------------------------------
    # Clear all active tool state
    # ------------------------------------------------------------------

    def _clear_active(self):
        self.line_start = None
        self.polyline_start = None
        self.circle_center = None
        self.rect_start = None
        self.arc_pts = []
        self.ellipse_pts = []
        self.regpoly_center = None
        self.spline_pts = []

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        self.steps += 1
        x, y = int(action["x"]), int(action["y"])
        tool = int(action["tool"])
        click = bool(action["click"] > 0.5)
        snap = bool(action["snap"] > 0.5)
        end_sess = bool(action["end"] > 0.5)
        self.cursor_pos = (x, y)
        self.ghost_layer.fill(0)

        # --- Line tool ---
        if tool == TOOL_MAP["Line"]:
            if self.line_start is None:
                if click:
                    self.line_start = (x, y)
                    self.polyline_start = (x, y)
            else:
                rasterize_line(self.ghost_layer, self.line_start, (x, y), 0.4, 1)
                if click:
                    end_pt = self.polyline_start if (snap and self.polyline_start) else (x, y)
                    rasterize_line(self.drawing_layer, self.line_start, end_pt, 1.0, 2)
                    if snap:
                        self.line_start = None
                        self.polyline_start = None
                    else:
                        self.line_start = end_pt

        # --- Circle tool ---
        elif tool == TOOL_MAP["Circle"]:
            if self.circle_center is None:
                if click:
                    self.circle_center = (x, y)
            else:
                radius = np.linalg.norm(np.array([x, y]) - np.array(self.circle_center))
                rasterize_circle(self.ghost_layer, self.circle_center, radius, 0.4, 1)
                if click:
                    rasterize_circle(self.drawing_layer, self.circle_center, radius, 1.0, 2)
                    self.circle_center = None

        # --- Rectangle tool ---
        elif tool == TOOL_MAP["Rectangle"]:
            if self.rect_start is None:
                if click:
                    self.rect_start = (x, y)
            else:
                rasterize_rectangle(self.ghost_layer, self.rect_start, (x, y), 0.4, 1)
                if click:
                    rasterize_rectangle(self.drawing_layer, self.rect_start, (x, y), 1.0, 2)
                    self.rect_start = None

        # --- Arc tool (three-point) ---
        elif tool == TOOL_MAP["Arc"]:
            if click:
                self.arc_pts.append((x, y))
                if len(self.arc_pts) == 3:
                    rasterize_arc(self.drawing_layer,
                                  self.arc_pts[0], self.arc_pts[1], self.arc_pts[2],
                                  1.0, 2)
                    self.arc_pts = []

        # --- Ellipse tool ---
        elif tool == TOOL_MAP["Ellipse"]:
            if click:
                self.ellipse_pts.append((x, y))
                if len(self.ellipse_pts) == 3:
                    c = self.ellipse_pts[0]
                    ep_a = self.ellipse_pts[1]
                    ep_b = self.ellipse_pts[2]
                    a_len = np.sqrt((ep_a[0] - c[0]) ** 2 + (ep_a[1] - c[1]) ** 2)
                    b_len = np.sqrt((ep_b[0] - c[0]) ** 2 + (ep_b[1] - c[1]) ** 2)
                    angle = np.degrees(np.arctan2(ep_a[1] - c[1], ep_a[0] - c[0]))
                    rasterize_ellipse(self.drawing_layer, c, (a_len, b_len), angle, 1.0, 2)
                    self.ellipse_pts = []
            elif len(self.ellipse_pts) == 1:
                c = self.ellipse_pts[0]
                r_ghost = np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2)
                rasterize_circle(self.ghost_layer, c, r_ghost, 0.4, 1)

        # --- RegPolygon tool ---
        elif tool == TOOL_MAP["RegPolygon"]:
            if self.regpoly_center is None:
                if click:
                    self.regpoly_center = (x, y)
            else:
                r = np.linalg.norm(np.array([x, y]) - np.array(self.regpoly_center))
                angle_off = np.arctan2(y - self.regpoly_center[1],
                                       x - self.regpoly_center[0])
                rasterize_regular_polygon(self.ghost_layer, self.regpoly_center,
                                           r, 6, angle_off, 0.4, 1)
                if click:
                    rasterize_regular_polygon(self.drawing_layer, self.regpoly_center,
                                               r, 6, angle_off, 1.0, 2)
                    self.regpoly_center = None

        # --- Spline tool (Bezier) ---
        elif tool == TOOL_MAP["Spline"]:
            if click:
                self.spline_pts.append((x, y))
                if snap or len(self.spline_pts) >= 8:
                    if len(self.spline_pts) >= 2:
                        rasterize_bezier(self.drawing_layer, self.spline_pts, 1.0, 2)
                    self.spline_pts = []
            elif len(self.spline_pts) >= 1:
                # Ghost: preview of current spline
                preview = list(self.spline_pts) + [(x, y)]
                if len(preview) >= 2:
                    rasterize_bezier(self.ghost_layer, preview, 0.4, 1)

        # --- Any other tool clears active state ---
        else:
            self._clear_active()

        self.prev_tool = tool
        self.prev_click = 1.0 if click else 0.0

        # --- Episode termination and reward ---
        done = end_sess or self.steps >= self.max_steps
        reward = 0.0
        info = {}
        if done:
            iou = thick_line_iou(self.target_layer, self.drawing_layer,
                                 thickness=self.reward_thickness)
            info["iou"] = iou
            reward = iou * 100.0 if end_sess else (iou * 100.0) - 10.0

        # Small step penalty
        reward -= 0.1

        return self._get_obs(), reward, done, info
