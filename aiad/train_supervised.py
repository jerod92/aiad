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

from aiad.checkpoint import BestModelTracker, load_checkpoint
from aiad.config import DEVICE, NUM_TOOLS, MODEL_PRESETS
from aiad.dataset import MixedShapeDataset
from aiad.model import CADModel
from aiad.viz import visualize_dataset_samples, visualize_predictions


def train(
    epochs=1,
    batch_size=8,
    lr=3e-4,
    num_samples=16_000,
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
        is_best = tracker.update(model, optimizer, avg_loss,
                                 metadata={"epoch": epoch + 1, "global_step": global_step,
                                            "model_size": model_size})
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
    p.add_argument("--num-samples", type=int, default=16_000)
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
