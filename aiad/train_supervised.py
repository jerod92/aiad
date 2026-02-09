"""
Supervised pre-training: train the model to predict the correct next action
given a random mid-drawing state of a synthetic shape.

Supports ``--model-size large|mini`` to select the model preset.

Usage:
    python -m aiad.train_supervised [OPTIONS]

Or from a notebook:
    from aiad.train_supervised import train
    train(epochs=5, batch_size=16, model_size="mini")
"""

import argparse
import signal
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from aiad.checkpoint import BestModelTracker, load_checkpoint
from aiad.config import DEVICE, NUM_TOOLS, MODEL_PRESETS, TOOLS, TOOL_MAP
from aiad.dataset import MixedShapeDataset
from aiad.model import CADModel
from aiad.shapes import random_shape
from aiad.viz import visualize_dataset_samples, visualize_predictions


def sequential_accuracy_test(model, device, img_size, n_shapes=100,
                             pos_tol=15):
    """Test how many consecutive steps the model predicts correctly.

    For each shape, replay the action sequence from step 0.  At each step,
    feed the *ground-truth* partial drawing state and check whether the
    model's prediction matches the target action:
      - tool must match exactly
      - click/snap/end must match (binary)
      - (x, y) must be within ``pos_tol`` pixels

    Returns
    -------
    dict with keys:
        avg_consecutive : float — average longest consecutive correct streak
        avg_total_correct : float — average fraction of steps correct
        avg_seq_len : float — average action sequence length
    """
    from aiad.dataset import MixedShapeDataset
    from aiad.raster import draw_cursor_np, gaussian_blur, rasterize_line

    ds = MixedShapeDataset(num_samples=1, img_size=img_size)
    model.eval()

    consecutive_streaks = []
    total_correct_fracs = []
    seq_lens = []

    for _ in range(n_shapes):
        shape = random_shape(img_size)
        actions = shape.actions
        if len(actions) < 2:
            continue

        S = img_size
        seq_lens.append(len(actions))

        # Build the full target image via replay
        target_layer = ds._replay_actions(actions, S)
        target_layer = gaussian_blur(target_layer)

        # Random appearance for target (same as dataset)
        bg_val, line_val = ds._random_appearance()
        target_display = np.full((S, S), bg_val, dtype=np.float32)
        mask = target_layer > 0.05
        target_display[mask] = line_val
        target_rgb = np.stack([target_display] * 3, axis=0)

        longest_streak = 0
        current_streak = 0
        total_correct = 0

        for step_idx in range(len(actions)):
            target_action = actions[step_idx]

            # Build partial drawing by replaying 0..step_idx-1
            if step_idx > 0:
                drawing = ds._replay_actions(actions[:step_idx], S)
                prev_tool = actions[step_idx - 1].tool
                prev_click = actions[step_idx - 1].click
            else:
                drawing = np.zeros((S, S), dtype=np.float32)
                prev_tool = TOOL_MAP["None"]
                prev_click = 0.0

            ghost = np.zeros((S, S), dtype=np.float32)
            cursor = np.zeros((S, S), dtype=np.float32)
            cx, cy = int(np.clip(target_action.x, 0, S - 1)), int(np.clip(target_action.y, 0, S - 1))
            draw_cursor_np(cursor, cx, cy)

            obs = np.concatenate([
                target_rgb,
                drawing[np.newaxis],
                ghost[np.newaxis],
                cursor[np.newaxis],
            ], axis=0)

            obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            pt_t = torch.tensor([prev_tool], dtype=torch.long, device=device)
            pc_t = torch.tensor([[prev_click]], dtype=torch.float32, device=device)

            with torch.no_grad():
                out = model(obs_t, pt_t, pc_t)
                pred_x = out["x"].argmax(dim=1).item()
                pred_y = out["y"].argmax(dim=1).item()
                pred_tool = out["tool"].argmax(dim=1).item()
                pred_click = int(torch.sigmoid(out["click"]).item() > 0.5)
                pred_snap = int(torch.sigmoid(out["snap"]).item() > 0.5)
                pred_end = int(torch.sigmoid(out["end"]).item() > 0.5)

            # Check correctness
            gt_x = int(np.clip(target_action.x, 0, S - 1))
            gt_y = int(np.clip(target_action.y, 0, S - 1))
            gt_click = int(target_action.click > 0.5)
            gt_snap = int(target_action.snap > 0.5)
            gt_end = int(target_action.end > 0.5)

            tool_ok = pred_tool == target_action.tool
            pos_ok = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) <= pos_tol
            click_ok = pred_click == gt_click
            snap_ok = pred_snap == gt_snap
            end_ok = pred_end == gt_end

            correct = tool_ok and pos_ok and click_ok and snap_ok and end_ok

            if correct:
                current_streak += 1
                total_correct += 1
            else:
                longest_streak = max(longest_streak, current_streak)
                current_streak = 0

        longest_streak = max(longest_streak, current_streak)
        consecutive_streaks.append(longest_streak)
        total_correct_fracs.append(total_correct / len(actions))

    result = {
        "avg_consecutive": float(np.mean(consecutive_streaks)) if consecutive_streaks else 0.0,
        "avg_total_correct": float(np.mean(total_correct_fracs)) if total_correct_fracs else 0.0,
        "avg_seq_len": float(np.mean(seq_lens)) if seq_lens else 0.0,
        "n_shapes": len(consecutive_streaks),
    }
    return result


def train(
    epochs=1,
    batch_size=8,
    lr=3e-4,
    num_samples=32_000,
    model_size="large",
    checkpoint_dir="checkpoints/supervised",
    output_dir="outputs/supervised",
    resume=None,
    viz_interval=500,
    log_interval=50,
    device=None,
):
    """Run supervised pre-training. Callable from scripts or notebooks."""
    device = device or DEVICE
    import os
    os.makedirs(output_dir, exist_ok=True)

    preset = MODEL_PRESETS[model_size]
    img_size = preset.img_size

    model = CADModel.from_preset(model_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tracker = BestModelTracker(checkpoint_dir, metric_name="loss", mode="min")
    start_epoch = 0

    if resume:
        meta = load_checkpoint(resume, model, optimizer, device)
        start_epoch = meta.get("epoch", 0)
        print(f"Resumed from {resume} (epoch {start_epoch})")

    dataset = MixedShapeDataset(num_samples=num_samples, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=torch.cuda.is_available())

    # Save dataset samples once for quality inspection
    visualize_dataset_samples(dataset, num_samples=16,
                              save_path=f"{output_dir}/dataset_samples.png")

    # Graceful interrupt handler
    interrupted = False
    def _handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nInterrupted — saving checkpoint and exiting ...")
    signal.signal(signal.SIGINT, _handler)

    model.train()
    global_step = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=True)

        for batch in pbar:
            if interrupted:
                break

            obs = batch["obs"].to(device)
            prev_tool = batch["prev_tool"].to(device)
            prev_click = batch["prev_click"].unsqueeze(1).to(device)
            t_x = batch["target_x"].to(device)
            t_y = batch["target_y"].to(device)
            t_tool = batch["target_tool"].to(device)
            t_click = batch["target_click"].unsqueeze(1).to(device)
            t_snap = batch["target_snap"].unsqueeze(1).to(device)
            t_end = batch["target_end"].unsqueeze(1).to(device)

            out = model(obs, prev_tool, prev_click)

            loss_x = F.cross_entropy(out["x"], t_x)
            loss_y = F.cross_entropy(out["y"], t_y)
            loss_tool = F.cross_entropy(out["tool"], t_tool)
            loss_click = F.binary_cross_entropy_with_logits(out["click"], t_click)
            loss_snap = F.binary_cross_entropy_with_logits(out["snap"], t_snap)
            loss_end = F.binary_cross_entropy_with_logits(out["end"], t_end)
            total = loss_x + loss_y + loss_tool + loss_click + loss_snap + loss_end

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            epoch_loss += total.item()
            epoch_steps += 1
            global_step += 1

            if global_step % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{total.item():.3f}",
                    "x": f"{loss_x.item():.2f}",
                    "y": f"{loss_y.item():.2f}",
                    "tool": f"{loss_tool.item():.2f}",
                })

            if global_step % viz_interval == 0:
                model.eval()
                visualize_predictions(
                    model, dataset, device, num_samples=8,
                    save_path=f"{output_dir}/pred_step_{global_step}.png",
                )
                model.train()

        if interrupted:
            break

        avg_loss = epoch_loss / max(epoch_steps, 1)

        # --- Post-epoch sequential accuracy test ---
        model.eval()
        print(f"Epoch {epoch + 1} — running sequential accuracy test (n=100) ...")
        seq_result = sequential_accuracy_test(model, device, img_size, n_shapes=100)
        print(f"  Seq accuracy: avg_consecutive={seq_result['avg_consecutive']:.1f}"
              f"  avg_correct={seq_result['avg_total_correct']:.3f}"
              f"  avg_seq_len={seq_result['avg_seq_len']:.1f}")
        model.train()

        is_best = tracker.update(model, optimizer, avg_loss,
                                 metadata={"epoch": epoch + 1, "global_step": global_step,
                                            "model_size": model_size,
                                            "seq_accuracy": seq_result})
        best_tag = " *best*" if is_best else ""
        print(f"Epoch {epoch + 1} done — avg loss: {avg_loss:.4f}{best_tag}")

    # Final save
    tracker.update(model, optimizer, avg_loss,
                   metadata={"epoch": epoch + 1, "global_step": global_step,
                              "model_size": model_size, "final": True})
    print(f"Training complete. Checkpoints in {checkpoint_dir}/")
    return model


def main():
    p = argparse.ArgumentParser(description="Supervised pre-training for CAD tracer")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-samples", type=int, default=32_000)
    p.add_argument("--model-size", choices=["large", "mini"], default="large",
                   help="Model preset: 'large' (512px) or 'mini' (256px, Kaggle-friendly)")
    p.add_argument("--checkpoint-dir", default="checkpoints/supervised")
    p.add_argument("--output-dir", default="outputs/supervised")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--viz-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=50)
    args = p.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
