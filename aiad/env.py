"""
CAD drawing environment for PPO training.

The agent sees a target raster image and must reproduce it using CAD tools
(Line, Circle).  At episode end, reward = thick-line IoU between the agent's
drawing and the target.
"""

import numpy as np
import cv2

from aiad.config import TOOL_MAP, IMG_SIZE
from aiad.raster import rasterize_line, rasterize_circle, draw_cursor_np, gaussian_blur


def _random_points(num_points, margin, img_size, min_dist=50):
    points = [np.random.randint(margin, img_size - margin, 2)]
    for _ in range(num_points - 1):
        pt = np.random.randint(margin, img_size - margin, 2)
        while len(points) > 0 and np.linalg.norm(pt - points[-1]) < min_dist:
            pt = np.random.randint(margin, img_size - margin, 2)
        points.append(pt)
    return points


def _canonical_order(points):
    arr = np.array(points)
    min_idx = np.lexsort((arr[:, 0], arr[:, 1]))[0]
    return points[min_idx:] + points[:min_idx]


def thick_line_iou(target_layer, drawing_layer, target_points, is_polygon,
                   target_center, target_radius, thickness=10):
    """Compute IoU between target and drawing using dilated (thick) masks.

    Both the ground-truth shape and the model's drawing are thickened so that
    near-miss parallel lines still receive partial credit.
    """
    S = target_layer.shape[0]

    # Thick target mask
    target_mask = np.zeros((S, S), dtype=np.float32)
    if target_center is not None:
        rasterize_circle(target_mask, target_center, target_radius, 1.0, thickness)
    else:
        n = len(target_points)
        segs = n if is_polygon else n - 1
        for i in range(segs):
            a = target_points[i]
            b = target_points[(i + 1) % n]
            rasterize_line(target_mask, a, b, 1.0, thickness)

    # Thick drawing mask
    kernel = np.ones((thickness, thickness), np.uint8)
    drawn_mask = cv2.dilate(drawing_layer, kernel, iterations=1)

    intersection = np.logical_and(target_mask > 0.5, drawn_mask > 0.5).sum()
    union = np.logical_or(target_mask > 0.5, drawn_mask > 0.5).sum()
    return float(intersection / (union + 1e-6))


class MixedCADEnvironment:
    """Gym-like environment for CAD tracing with mixed shape targets."""

    def __init__(self, img_size=IMG_SIZE, max_steps=30, reward_thickness=10):
        self.img_size = img_size
        self.max_steps = max_steps
        self.reward_thickness = reward_thickness
        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        S = self.img_size
        margin = 30
        shape_type = np.random.choice(["line", "polyline", "polygon", "circle"])

        self.target_layer = np.zeros((S, S), dtype=np.float32)
        self.is_polygon = False
        self.t_points = None
        self.t_center = None
        self.t_radius = None

        if shape_type == "circle":
            safe = margin + 50
            self.t_center = np.random.randint(safe, S - safe, 2)
            max_r = min(self.t_center[0] - margin, S - self.t_center[0] - margin,
                        self.t_center[1] - margin, S - self.t_center[1] - margin)
            self.t_radius = np.random.randint(50, max_r + 1)
            rasterize_circle(self.target_layer, self.t_center, self.t_radius, 1.0, 3)
        else:
            if shape_type == "line":
                num_seg, self.is_polygon = 1, False
            elif shape_type == "polyline":
                num_seg, self.is_polygon = np.random.randint(2, 7), False
            else:
                num_seg, self.is_polygon = np.random.randint(3, 7), True
            num_pts = num_seg if self.is_polygon else num_seg + 1
            points = _random_points(num_pts, margin, S)
            if self.is_polygon:
                while np.linalg.norm(points[0] - points[-1]) < 50:
                    points[-1] = np.random.randint(margin, S - margin, 2)
            self.t_points = _canonical_order(points)
            segs = num_seg if not self.is_polygon else num_seg
            for i in range(segs):
                a = self.t_points[i]
                b = (self.t_points[(i + 1) % num_pts] if self.is_polygon
                     else self.t_points[i + 1])
                rasterize_line(self.target_layer, a, b, 1.0, 3)

        self.target_layer = gaussian_blur(self.target_layer)
        self.drawing_layer = np.zeros((S, S), dtype=np.float32)
        self.ghost_layer = np.zeros((S, S), dtype=np.float32)
        self.cursor_pos = (S // 2, S // 2)

        self.active_line_start = None
        self.polyline_start = None
        self.active_circle_center = None
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

        # --- Line tool logic ---
        if tool == TOOL_MAP["Line"]:
            if self.active_line_start is None:
                if click:
                    self.active_line_start = (x, y)
                    self.polyline_start = (x, y)
            else:
                rasterize_line(self.ghost_layer, self.active_line_start, (x, y), 0.5, 1)
                if click:
                    end_pt = self.polyline_start if (snap and self.polyline_start) else (x, y)
                    rasterize_line(self.drawing_layer, self.active_line_start, end_pt, 1.0, 1)
                    if snap:
                        self.active_line_start = None
                        self.polyline_start = None
                    else:
                        self.active_line_start = end_pt

        # --- Circle tool logic ---
        elif tool == TOOL_MAP["Circle"]:
            if self.active_circle_center is None:
                if click:
                    self.active_circle_center = (x, y)
            else:
                radius = np.linalg.norm(np.array([x, y]) - np.array(self.active_circle_center))
                rasterize_circle(self.ghost_layer, self.active_circle_center, radius, 0.5, 1)
                if click:
                    rasterize_circle(self.drawing_layer, self.active_circle_center, radius, 1.0, 1)
                    self.active_circle_center = None

        # --- Any other tool clears active state ---
        else:
            self.active_line_start = None
            self.polyline_start = None
            self.active_circle_center = None

        self.prev_tool = tool
        self.prev_click = 1.0 if click else 0.0

        # --- Episode termination and reward ---
        done = end_sess or self.steps >= self.max_steps
        reward = 0.0
        info = {}
        if done:
            iou = thick_line_iou(
                self.target_layer, self.drawing_layer,
                self.t_points, self.is_polygon,
                self.t_center, self.t_radius,
                thickness=self.reward_thickness,
            )
            info["iou"] = iou
            # Bonus for voluntary end, penalty for timeout
            reward = iou * 100.0 if end_sess else (iou * 100.0) - 10.0

        # Small step penalty to encourage efficiency
        reward -= 0.1

        return self._get_obs(), reward, done, info
