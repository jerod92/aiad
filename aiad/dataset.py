"""
Synthetic dataset for supervised pre-training.

Generates random shapes (line, polyline, polygon, circle), simulates a random
mid-drawing state, and provides the correct next action as the training target.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from aiad.config import TOOL_MAP, IMG_SIZE
from aiad.raster import rasterize_line, rasterize_circle, draw_cursor_np, gaussian_blur


def _random_points(num_points, margin, img_size, min_dist=50):
    """Generate *num_points* well-separated random points inside the image."""
    points = [np.random.randint(margin, img_size - margin, 2)]
    for _ in range(num_points - 1):
        pt = np.random.randint(margin, img_size - margin, 2)
        while len(points) > 0 and np.linalg.norm(pt - points[-1]) < min_dist:
            pt = np.random.randint(margin, img_size - margin, 2)
        points.append(pt)
    return points


def _canonical_order(points):
    """Rotate point list so that the lexicographically smallest point is first."""
    arr = np.array(points)
    min_idx = np.lexsort((arr[:, 0], arr[:, 1]))[0]
    return points[min_idx:] + points[:min_idx]


class MixedShapeDataset(Dataset):
    """On-the-fly generator of supervised training samples.

    Each sample contains:
      obs          [6, H, W] — target (3ch RGB) + drawing + ghost + cursor
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
    # Circle samples
    # ------------------------------------------------------------------

    def _make_circle(self):
        S = self.img_size
        margin = 20
        safe = margin + 50
        center = np.random.randint(safe, S - safe, 2)
        max_r = min(center[0] - margin, S - center[0] - margin,
                    center[1] - margin, S - center[1] - margin)
        radius = np.random.randint(50, max_r + 1)
        top_pt = np.array([center[0], center[1] - radius])

        target_layer = np.zeros((S, S), dtype=np.float32)
        rasterize_circle(target_layer, center, radius, 1.0, 3)
        target_layer = gaussian_blur(target_layer)
        target_rgb = np.stack([target_layer] * 3, axis=0)

        drawing = np.zeros((S, S), dtype=np.float32)
        ghost = np.zeros((S, S), dtype=np.float32)
        cursor = np.zeros((S, S), dtype=np.float32)
        cx, cy = np.random.randint(0, S, 2)
        draw_cursor_np(cursor, cx, cy)

        if np.random.rand() < 0.5:  # beginning: move to center
            return self._pack(
                target_rgb, drawing, ghost, cursor,
                prev_tool=TOOL_MAP["None"], prev_click=0.0,
                tx=int(center[0]), ty=int(center[1]),
                ttool=TOOL_MAP["Circle"], tclick=1.0, tsnap=0.0, tend=0.0,
            )
        else:  # end: click radius point
            rand_r = np.linalg.norm(np.array([cx, cy]) - center)
            rasterize_circle(ghost, center, rand_r, 0.5, 1)
            return self._pack(
                target_rgb, drawing, ghost, cursor,
                prev_tool=TOOL_MAP["Circle"], prev_click=0.0,
                tx=int(top_pt[0]), ty=int(top_pt[1]),
                ttool=TOOL_MAP["Circle"], tclick=1.0, tsnap=0.0, tend=1.0,
            )

    # ------------------------------------------------------------------
    # Line / polyline / polygon samples
    # ------------------------------------------------------------------

    def _make_line_shape(self, shape_type):
        S = self.img_size
        margin = 20
        if shape_type == "line":
            num_seg, is_poly = 1, False
        elif shape_type == "polyline":
            num_seg, is_poly = np.random.randint(2, 7), False
        else:
            num_seg, is_poly = np.random.randint(3, 7), True

        num_pts = num_seg if is_poly else num_seg + 1
        points = _random_points(num_pts, margin, S)

        if is_poly:
            while np.linalg.norm(points[0] - points[-1]) < 50:
                points[-1] = np.random.randint(margin, S - margin, 2)
        points = _canonical_order(points)

        # Rasterize target
        target_layer = np.zeros((S, S), dtype=np.float32)
        for i in range(num_seg):
            a = points[i]
            b = points[(i + 1) % num_pts] if is_poly else points[i + 1]
            rasterize_line(target_layer, a, b, 1.0, 3)
        target_layer = gaussian_blur(target_layer)
        target_rgb = np.stack([target_layer] * 3, axis=0)

        drawing = np.zeros((S, S), dtype=np.float32)
        ghost = np.zeros((S, S), dtype=np.float32)
        cursor = np.zeros((S, S), dtype=np.float32)
        cx, cy = np.random.randint(0, S, 2)
        draw_cursor_np(cursor, cx, cy)

        # Pick scenario
        if num_seg == 1:
            scenario = "beginning" if np.random.rand() < 0.5 else "final"
        else:
            r = np.random.rand()
            scenario = "beginning" if r < 1 / 3 else ("mid" if r < 2 / 3 else "final")

        if scenario == "beginning":
            return self._pack(
                target_rgb, drawing, ghost, cursor,
                prev_tool=TOOL_MAP["None"], prev_click=0.0,
                tx=int(points[0][0]), ty=int(points[0][1]),
                ttool=TOOL_MAP["Line"], tclick=1.0, tsnap=0.0, tend=0.0,
            )

        # mid or final: draw some segments already
        if scenario == "mid":
            seg_idx = np.random.randint(1, num_seg)
            t_end, t_snap = 0.0, 0.0
        else:
            seg_idx = num_seg
            t_end = 1.0
            t_snap = 1.0 if is_poly else 0.0

        for i in range(seg_idx - 1):
            rasterize_line(drawing, points[i], points[i + 1], 1.0, 1)

        active_start = points[seg_idx - 1]
        rasterize_line(ghost, active_start, (cx, cy), 0.5, 1)

        target_idx = 0 if (is_poly and scenario == "final") else seg_idx
        tx, ty = int(points[target_idx][0]), int(points[target_idx][1])
        return self._pack(
            target_rgb, drawing, ghost, cursor,
            prev_tool=TOOL_MAP["Line"], prev_click=0.0,
            tx=tx, ty=ty,
            ttool=TOOL_MAP["Line"], tclick=1.0, tsnap=t_snap, tend=t_end,
        )

    # ------------------------------------------------------------------
    # Helpers
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

    def __getitem__(self, idx):
        shape = np.random.choice(["line", "polyline", "polygon", "circle"])
        if shape == "circle":
            return self._make_circle()
        return self._make_line_shape(shape)
