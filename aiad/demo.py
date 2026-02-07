"""
Interactive tracing demo.

Load a trained model and let it trace an uploaded image or a blank canvas
in real-time.  Produces an animated GIF of the tracing process.

Usage (CLI):
    python -m aiad.demo --checkpoint checkpoints/ppo/best.pt --image photo.png

Or from a notebook:
    from aiad.demo import trace_image
    trace_image("checkpoints/ppo/best.pt", image_path="photo.png")
    trace_image("checkpoints/ppo/best.pt")  # random shape target
"""

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

from aiad.checkpoint import load_checkpoint
from aiad.config import DEVICE, MODEL_PRESETS, NUM_TOOLS
from aiad.model import CADModel
from aiad.raster import draw_cursor_np, gaussian_blur
from aiad.viz import _make_frame


def _load_target(image_path, img_size):
    """Load and preprocess an external image as a grayscale target."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # Invert if mostly white (assume dark lines on light bg)
    if img.mean() > 128:
        img = 255 - img
    # Edge detection for clean outlines
    edges = cv2.Canny(img, 50, 150)
    target = edges.astype(np.float32) / 255.0
    target = gaussian_blur(target)
    return target


def trace_image(
    checkpoint_path,
    image_path=None,
    model_size="large",
    max_steps=80,
    deterministic=True,
    save_path="outputs/demo_trace.gif",
    device=None,
):
    """Load a trained model and trace a target image.

    Parameters
    ----------
    checkpoint_path : str
        Path to trained model checkpoint (.pt file).
    image_path : str or None
        Path to input image.  If None, uses a random synthetic shape.
    model_size : str
        Model preset name ('large' or 'mini').
    max_steps : int
        Maximum number of drawing steps.
    deterministic : bool
        If True, use greedy actions.  False for stochastic.
    save_path : str
        Where to save the output GIF.
    device : torch.device or None

    Returns
    -------
    list of PIL.Image
        The rendered frames.
    """
    device = device or DEVICE
    preset = MODEL_PRESETS[model_size]
    img_size = preset.img_size

    # Build & load model
    model = CADModel.from_preset(model_size).to(device)
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()

    # Prepare target
    if image_path is not None:
        target_layer = _load_target(image_path, img_size)
    else:
        from aiad.shapes import random_shape
        shape = random_shape(img_size)
        target_layer = shape.target_layer

    S = img_size
    drawing = np.zeros((S, S), dtype=np.float32)
    ghost = np.zeros((S, S), dtype=np.float32)
    cursor_pos = (S // 2, S // 2)
    prev_tool = 0
    prev_click = 0.0
    frames = []

    # Tool active state (mirroring env)
    line_start = None
    polyline_start = None
    circle_center = None
    rect_start = None
    arc_pts = []
    ellipse_pts = []
    regpoly_center = None
    spline_pts = []

    from aiad.config import TOOL_MAP
    from aiad.raster import (
        rasterize_line, rasterize_circle, rasterize_rectangle,
        rasterize_ellipse, rasterize_arc, rasterize_bezier,
        rasterize_regular_polygon,
    )

    for step in range(max_steps):
        # Build observation
        target_rgb = np.stack([target_layer] * 3, axis=0)
        cursor = np.zeros((S, S), dtype=np.float32)
        cx, cy = cursor_pos
        draw_cursor_np(cursor, cx, cy)
        obs = np.concatenate([
            target_rgb,
            drawing[np.newaxis],
            ghost[np.newaxis],
            cursor[np.newaxis],
        ], axis=0)

        # Render frame
        obs_tuple = (obs, prev_tool, prev_click)
        frames.append(_make_frame_demo(obs, S))

        # Model forward
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        pt_t = torch.tensor([prev_tool], dtype=torch.long, device=device)
        pc_t = torch.tensor([[prev_click]], dtype=torch.float32, device=device)

        with torch.no_grad():
            action, _, _, _ = model.get_action(obs_t, pt_t, pc_t,
                                                deterministic=deterministic)

        x = action["x"].item()
        y = action["y"].item()
        tool = action["tool"].item()
        click = action["click"].item() > 0.5
        snap = action["snap"].item() > 0.5
        end_sess = action["end"].item() > 0.5

        cursor_pos = (x, y)
        ghost.fill(0)

        # Execute tool (same logic as env.py)
        if tool == TOOL_MAP["Line"]:
            if line_start is None:
                if click:
                    line_start = (x, y)
                    polyline_start = (x, y)
            else:
                rasterize_line(ghost, line_start, (x, y), 0.4, 1)
                if click:
                    end_pt = polyline_start if (snap and polyline_start) else (x, y)
                    rasterize_line(drawing, line_start, end_pt, 1.0, 2)
                    if snap:
                        line_start = None
                        polyline_start = None
                    else:
                        line_start = end_pt
        elif tool == TOOL_MAP["Circle"]:
            if circle_center is None:
                if click:
                    circle_center = (x, y)
            else:
                r = np.linalg.norm(np.array([x, y]) - np.array(circle_center))
                rasterize_circle(ghost, circle_center, r, 0.4, 1)
                if click:
                    rasterize_circle(drawing, circle_center, r, 1.0, 2)
                    circle_center = None
        elif tool == TOOL_MAP["Rectangle"]:
            if rect_start is None:
                if click:
                    rect_start = (x, y)
            else:
                rasterize_rectangle(ghost, rect_start, (x, y), 0.4, 1)
                if click:
                    rasterize_rectangle(drawing, rect_start, (x, y), 1.0, 2)
                    rect_start = None
        elif tool == TOOL_MAP["Spline"]:
            if click:
                spline_pts.append((x, y))
                if snap or len(spline_pts) >= 8:
                    if len(spline_pts) >= 2:
                        rasterize_bezier(drawing, spline_pts, 1.0, 2)
                    spline_pts = []
            elif len(spline_pts) >= 1:
                preview = list(spline_pts) + [(x, y)]
                if len(preview) >= 2:
                    rasterize_bezier(ghost, preview, 0.4, 1)

        prev_tool = tool
        prev_click = 1.0 if click else 0.0

        if end_sess:
            break

    # Final frame
    target_rgb = np.stack([target_layer] * 3, axis=0)
    cursor = np.zeros((S, S), dtype=np.float32)
    draw_cursor_np(cursor, cursor_pos[0], cursor_pos[1])
    obs = np.concatenate([target_rgb, drawing[np.newaxis],
                          ghost[np.newaxis], cursor[np.newaxis]], axis=0)
    frames.append(_make_frame_demo(obs, S))

    # Save GIF
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if frames:
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                        duration=200, loop=0)
    print(f"Demo trace saved to {save_path} ({len(frames)} frames)")
    return frames


def _make_frame_demo(obs, S):
    """Build a 3-panel frame: target | overlay | drawing-only."""
    target = np.stack([obs[0]] * 3, axis=-1)
    overlay = obs[:3].transpose(1, 2, 0).copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + obs[3], 0, 1)
    overlay[..., 1] = np.clip(overlay[..., 1] + obs[4] * 0.6, 0, 1)
    overlay[..., 2] = np.clip(overlay[..., 2] + obs[5] * 0.8, 0, 1)

    canvas = np.zeros((S, S, 3), dtype=np.float32)
    canvas[..., 0] = np.clip(obs[3] + obs[4] * 0.5, 0, 1)
    canvas[..., 1] = np.clip(obs[4] * 0.5, 0, 1)
    canvas[..., 2] = np.clip(obs[5] * 0.8, 0, 1)

    sep = np.ones((S, 2, 3), dtype=np.float32)
    combined = np.concatenate([target, sep, overlay, sep, canvas], axis=1)
    return Image.fromarray((combined * 255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser(description="AIAD interactive tracing demo")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--image", default=None, help="Path to input image (optional)")
    p.add_argument("--model-size", choices=["large", "mini"], default="large")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic rather than greedy actions")
    p.add_argument("--save-path", default="outputs/demo_trace.gif")
    args = p.parse_args()
    trace_image(
        args.checkpoint,
        image_path=args.image,
        model_size=args.model_size,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
