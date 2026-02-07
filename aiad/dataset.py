"""
Synthetic dataset for supervised pre-training.

Generates random shapes via the ``aiad.shapes`` package, simulates a random
mid-drawing state, and provides the correct next action as the training target.

Supports both *large* (512 px) and *mini* (256 px) image sizes.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from aiad.config import TOOL_MAP, IMG_SIZE
from aiad.raster import (
    rasterize_line, rasterize_circle, rasterize_rectangle,
    rasterize_ellipse, rasterize_bezier, rasterize_arc,
    rasterize_regular_polygon, draw_cursor_np, gaussian_blur,
)
from aiad.shapes import ALL_GENERATORS, random_shape, ShapeSample, ActionStep


class MixedShapeDataset(Dataset):
    """On-the-fly generator of supervised training samples.

    Each sample contains:
      obs          [6, H, W] — target (3ch grey→RGB) + drawing + ghost + cursor
      prev_tool    long       — tool index at the current state
      prev_click   float      — whether the previous action was a click
      target_x     long       — correct next X position
      target_y     long       — correct next Y position
      target_tool  long       — correct next tool
      target_click float      — whether the next action should click
      target_snap  float      — whether the next action should snap
      target_end   float      — whether the drawing should end after this action
    """

    def __init__(self, num_samples=16_000, img_size=IMG_SIZE):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        shape = random_shape(self.img_size)
        return self._simulate_mid_drawing(shape)

    # ------------------------------------------------------------------
    # Simulate a mid-drawing state
    # ------------------------------------------------------------------

    def _simulate_mid_drawing(self, shape: ShapeSample):
        """Pick a random point in the action sequence and build a training
        sample from that partial state."""
        S = self.img_size
        actions = shape.actions
        n_actions = len(actions)

        if n_actions == 0:
            return self._fallback_line()

        # Pick scenario: beginning, mid, or final
        if n_actions <= 2:
            step_idx = 0 if np.random.rand() < 0.5 else n_actions - 1
        else:
            r = np.random.rand()
            if r < 0.33:
                step_idx = 0
            elif r < 0.67:
                step_idx = np.random.randint(1, n_actions - 1) if n_actions > 2 else 1
            else:
                step_idx = n_actions - 1

        target_action = actions[step_idx]

        # Build the drawing state by replaying actions 0..step_idx-1
        drawing = np.zeros((S, S), dtype=np.float32)
        ghost = np.zeros((S, S), dtype=np.float32)
        prev_tool = TOOL_MAP["None"]
        prev_click = 0.0

        if step_idx > 0:
            drawing = self._replay_actions(actions[:step_idx], S)
            prev_tool = actions[step_idx - 1].tool
            prev_click = actions[step_idx - 1].click

        # Ghost: line from last anchor to random cursor position
        cursor = np.zeros((S, S), dtype=np.float32)
        cx, cy = np.random.randint(0, S, 2)
        draw_cursor_np(cursor, cx, cy)

        if step_idx > 0 and actions[step_idx - 1].click > 0.5:
            last = actions[step_idx - 1]
            rasterize_line(ghost, (last.x, last.y), (cx, cy), 0.4, 1)

        # Target image (the full shape)
        target_layer = shape.target_layer
        target_rgb = np.stack([target_layer] * 3, axis=0)

        return self._pack(
            target_rgb, drawing, ghost, cursor,
            prev_tool=prev_tool, prev_click=prev_click,
            tx=int(np.clip(target_action.x, 0, S - 1)),
            ty=int(np.clip(target_action.y, 0, S - 1)),
            ttool=target_action.tool,
            tclick=target_action.click,
            tsnap=target_action.snap,
            tend=target_action.end,
        )

    # ------------------------------------------------------------------
    # Replay actions to build drawing state
    # ------------------------------------------------------------------

    def _replay_actions(self, actions, S):
        """Rasterise completed actions onto a drawing layer."""
        drawing = np.zeros((S, S), dtype=np.float32)
        line_start = None
        rect_start = None
        arc_pts = []
        circle_center = None
        ellipse_pts = []
        regpoly_center = None
        spline_pts = []

        for a in actions:
            pos = (a.x, a.y)
            if a.click < 0.5:
                continue

            tool = a.tool

            if tool == TOOL_MAP["Line"]:
                if line_start is not None:
                    rasterize_line(drawing, line_start, pos, 1.0, 2)
                line_start = pos if a.snap < 0.5 else None

            elif tool == TOOL_MAP["Circle"]:
                if circle_center is None:
                    circle_center = pos
                else:
                    r = np.linalg.norm(np.array(pos) - np.array(circle_center))
                    rasterize_circle(drawing, circle_center, r, 1.0, 2)
                    circle_center = None

            elif tool == TOOL_MAP["Rectangle"]:
                if rect_start is None:
                    rect_start = pos
                else:
                    rasterize_rectangle(drawing, rect_start, pos, 1.0, 2)
                    rect_start = None

            elif tool == TOOL_MAP["Arc"]:
                arc_pts.append(pos)
                if len(arc_pts) == 3:
                    rasterize_arc(drawing, arc_pts[0], arc_pts[1], arc_pts[2], 1.0, 2)
                    arc_pts = []

            elif tool == TOOL_MAP["Ellipse"]:
                ellipse_pts.append(pos)
                if len(ellipse_pts) == 3:
                    c = ellipse_pts[0]
                    ep_a = ellipse_pts[1]
                    ep_b = ellipse_pts[2]
                    a_len = np.sqrt((ep_a[0] - c[0]) ** 2 + (ep_a[1] - c[1]) ** 2)
                    b_len = np.sqrt((ep_b[0] - c[0]) ** 2 + (ep_b[1] - c[1]) ** 2)
                    angle = np.degrees(np.arctan2(ep_a[1] - c[1], ep_a[0] - c[0]))
                    rasterize_ellipse(drawing, c, (a_len, b_len), angle, 1.0, 2)
                    ellipse_pts = []

            elif tool == TOOL_MAP["RegPolygon"]:
                if regpoly_center is None:
                    regpoly_center = pos
                else:
                    r = np.linalg.norm(np.array(pos) - np.array(regpoly_center))
                    angle_off = np.arctan2(pos[1] - regpoly_center[1],
                                           pos[0] - regpoly_center[0])
                    # Default to hexagon when side count is unknown from replay
                    rasterize_regular_polygon(drawing, regpoly_center, r, 6,
                                              angle_off, 1.0, 2)
                    regpoly_center = None

            elif tool == TOOL_MAP["Spline"]:
                spline_pts.append(pos)
                if a.snap > 0.5 or a.end > 0.5:
                    if len(spline_pts) >= 2:
                        rasterize_bezier(drawing, spline_pts, 1.0, 2)
                    spline_pts = []

        return drawing

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_line(self):
        S = self.img_size
        m = 30
        p1 = np.random.randint(m, S - m, 2)
        p2 = np.random.randint(m, S - m, 2)
        target = np.zeros((S, S), dtype=np.float32)
        rasterize_line(target, p1, p2, 1.0, 2)
        target = gaussian_blur(target)
        target_rgb = np.stack([target] * 3, axis=0)
        drawing = np.zeros((S, S), dtype=np.float32)
        ghost = np.zeros((S, S), dtype=np.float32)
        cursor = np.zeros((S, S), dtype=np.float32)
        cx, cy = np.random.randint(0, S, 2)
        draw_cursor_np(cursor, cx, cy)
        return self._pack(
            target_rgb, drawing, ghost, cursor,
            prev_tool=TOOL_MAP["None"], prev_click=0.0,
            tx=int(p1[0]), ty=int(p1[1]),
            ttool=TOOL_MAP["Line"], tclick=1.0, tsnap=0.0, tend=0.0,
        )

    # ------------------------------------------------------------------
    # Pack sample
    # ------------------------------------------------------------------

    def _pack(self, target_rgb, drawing, ghost, cursor, *,
              prev_tool, prev_click, tx, ty, ttool, tclick, tsnap, tend):
        obs = np.concatenate([
            target_rgb,
            drawing[np.newaxis],
            ghost[np.newaxis],
            cursor[np.newaxis],
        ], axis=0)
        return {
            "obs": torch.from_numpy(obs),
            "prev_tool": torch.tensor(prev_tool, dtype=torch.long),
            "prev_click": torch.tensor(prev_click, dtype=torch.float32),
            "target_x": torch.tensor(tx, dtype=torch.long),
            "target_y": torch.tensor(ty, dtype=torch.long),
            "target_tool": torch.tensor(ttool, dtype=torch.long),
            "target_click": torch.tensor(tclick, dtype=torch.float32),
            "target_snap": torch.tensor(tsnap, dtype=torch.float32),
            "target_end": torch.tensor(tend, dtype=torch.float32),
        }
