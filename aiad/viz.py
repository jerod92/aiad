"""
Visualization and diagnostics toolkit.

Provides functions for inspecting dataset quality, watching the model draw,
plotting training curves, and comparing predictions to ground truth.

Usage (CLI):
    python -m aiad.viz dataset   [--num-samples 16] [--save-path ...]
    python -m aiad.viz episode   --checkpoint PATH  [--save-path ...]
    python -m aiad.viz curves    --checkpoint-dir DIR [--save-path ...]
    python -m aiad.viz predict   --checkpoint PATH  [--num-samples 8] [--save-path ...]
    python -m aiad.viz stats     [--num-samples 1000] [--save-path ...]

Or from a notebook:
    from aiad.viz import visualize_dataset_samples, create_episode_gif
    visualize_dataset_samples()
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from aiad.config import DEVICE, IMG_SIZE, NUM_TOOLS, TOOLS


# -----------------------------------------------------------------------
# 1. Dataset sample grid
# -----------------------------------------------------------------------

def visualize_dataset_samples(dataset=None, num_samples=16, save_path="outputs/dataset_samples.png"):
    """Render a grid of dataset samples showing target, drawing state, and action."""
    from aiad.dataset import MixedShapeDataset
    dataset = dataset or MixedShapeDataset(num_samples=max(num_samples, 64))

    cols = min(num_samples, 4)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(num_samples):
        sample = dataset[i]
        obs = sample["obs"].numpy()
        target = obs[:3].transpose(1, 2, 0)
        drawing = obs[3]
        ghost = obs[4]
        cursor = obs[5]

        # Composite: target in grey, drawing in red, ghost in green, cursor in blue
        display = target.copy()
        display[..., 0] = np.clip(display[..., 0] + drawing, 0, 1)
        display[..., 1] = np.clip(display[..., 1] + ghost * 0.6, 0, 1)
        display[..., 2] = np.clip(display[..., 2] + cursor * 0.8, 0, 1)

        ax = axes[i]
        ax.imshow(display)
        tx, ty = sample["target_x"].item(), sample["target_y"].item()
        ax.plot(tx, ty, "r+", markersize=12, markeredgewidth=2)
        tool_name = TOOLS[sample["target_tool"].item()]
        click = "click" if sample["target_click"].item() > 0.5 else ""
        snap = " snap" if sample["target_snap"].item() > 0.5 else ""
        end = " END" if sample["target_end"].item() > 0.5 else ""
        ax.set_title(f"{tool_name} {click}{snap}{end}\ntarget ({tx},{ty})", fontsize=9)
        ax.axis("off")

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Dataset samples saved to {save_path}")


# -----------------------------------------------------------------------
# 2. Episode GIF (watch the model draw)
# -----------------------------------------------------------------------

def create_episode_gif(env_or_none, model, device, save_path="outputs/episode.gif",
                       deterministic=True):
    """Roll out one episode and save a side-by-side GIF (target | canvas)."""
    from aiad.env import MixedCADEnvironment
    env = env_or_none or MixedCADEnvironment()
    obs_tuple = env.reset()
    frames = []
    done = False

    while not done:
        frame = _make_frame(env, obs_tuple)
        frames.append(frame)

        obs, pt, pc = obs_tuple
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        pt_t = torch.tensor([pt], dtype=torch.long, device=device)
        pc_t = torch.tensor([[pc]], dtype=torch.float32, device=device)

        with torch.no_grad():
            action, _, _, _ = model.get_action(obs_t, pt_t, pc_t, deterministic=deterministic)
        env_action = {k: v.item() for k, v in action.items()}
        obs_tuple, _, done, _ = env.step(env_action)

    # Final frame
    frames.append(_make_frame(env, obs_tuple))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if frames:
        frames[0].save(save_path, save_all=True, append_images=frames[1:],
                        duration=200, loop=0)
    print(f"Episode GIF saved to {save_path}")


def _make_frame(env, obs_tuple):
    """Build a 3-panel frame: target | overlay | drawing-only."""
    obs, _, _ = obs_tuple
    S = obs.shape[1]

    target = np.stack([obs[0]] * 3, axis=-1)
    overlay = obs[:3].transpose(1, 2, 0).copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + obs[3], 0, 1)
    overlay[..., 1] = np.clip(overlay[..., 1] + obs[4] * 0.6, 0, 1)
    overlay[..., 2] = np.clip(overlay[..., 2] + obs[5] * 0.8, 0, 1)

    canvas = np.zeros((S, S, 3), dtype=np.float32)
    canvas[..., 0] = np.clip(obs[3] + obs[4] * 0.5, 0, 1)
    canvas[..., 1] = np.clip(obs[4] * 0.5, 0, 1)
    canvas[..., 2] = np.clip(obs[5] * 0.8, 0, 1)

    # 2px white separator
    sep = np.ones((S, 2, 3), dtype=np.float32)
    combined = np.concatenate([target, sep, overlay, sep, canvas], axis=1)
    return Image.fromarray((combined * 255).astype(np.uint8))


# -----------------------------------------------------------------------
# 3. Training curves
# -----------------------------------------------------------------------

def plot_training_curves(history_path=None, checkpoint_dir=None,
                         save_path="outputs/training_curves.png"):
    """Plot loss / IoU / reward curves from a history.json file."""
    if history_path is None:
        if checkpoint_dir is None:
            raise ValueError("Provide history_path or checkpoint_dir")
        history_path = os.path.join(checkpoint_dir, "history.json")

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss (supervised)
    losses = [(i, h["loss"]) for i, h in enumerate(history) if "loss" in h]
    if losses:
        idx, vals = zip(*losses)
        axes[0].plot(idx, vals, linewidth=0.8)
        axes[0].set_title("Loss")
        axes[0].set_xlabel("update")

    # IoU (PPO)
    ious = [(i, h["avg_iou"]) for i, h in enumerate(history) if "avg_iou" in h]
    if ious:
        idx, vals = zip(*ious)
        axes[1].plot(idx, vals, color="green", linewidth=0.8)
        axes[1].set_title("Avg IoU")
        axes[1].set_xlabel("update")

    # Reward (PPO)
    rewards = [(i, h["avg_reward"]) for i, h in enumerate(history) if "avg_reward" in h]
    if rewards:
        idx, vals = zip(*rewards)
        axes[2].plot(idx, vals, color="orange", linewidth=0.8)
        axes[2].set_title("Avg Reward")
        axes[2].set_xlabel("update")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {save_path}")


# -----------------------------------------------------------------------
# 4. Model predictions vs ground truth
# -----------------------------------------------------------------------

def visualize_predictions(model, dataset=None, device=None, num_samples=8,
                          save_path="outputs/predictions.png"):
    """Show model's predicted actions alongside ground-truth targets."""
    from aiad.dataset import MixedShapeDataset
    device = device or DEVICE
    dataset = dataset or MixedShapeDataset(num_samples=max(num_samples, 64))

    model.eval()
    cols = min(num_samples, 4)
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(num_samples):
        sample = dataset[i]
        obs = sample["obs"].unsqueeze(0).to(device)
        pt = sample["prev_tool"].unsqueeze(0).to(device)
        pc = sample["prev_click"].unsqueeze(0).unsqueeze(1).to(device)

        with torch.no_grad():
            out = model(obs, pt, pc)
            pred_x = out["x"].argmax(dim=1).item()
            pred_y = out["y"].argmax(dim=1).item()
            pred_tool = out["tool"].argmax(dim=1).item()

        obs_np = sample["obs"].numpy()
        display = obs_np[:3].transpose(1, 2, 0).copy()
        display[..., 0] = np.clip(display[..., 0] + obs_np[3], 0, 1)
        display[..., 1] = np.clip(display[..., 1] + obs_np[4] * 0.6, 0, 1)

        ax = axes[i]
        ax.imshow(display)

        gt_x, gt_y = sample["target_x"].item(), sample["target_y"].item()
        ax.plot(gt_x, gt_y, "g+", markersize=14, markeredgewidth=2, label="GT")
        ax.plot(pred_x, pred_y, "rx", markersize=12, markeredgewidth=2, label="Pred")
        dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
        gt_tool = TOOLS[sample["target_tool"].item()]
        pd_tool = TOOLS[pred_tool]
        ax.set_title(f"GT: {gt_tool}@({gt_x},{gt_y})\nPred: {pd_tool}@({pred_x},{pred_y})  d={dist:.0f}",
                     fontsize=8)
        ax.axis("off")
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Predictions saved to {save_path}")


# -----------------------------------------------------------------------
# 5. Dataset statistics
# -----------------------------------------------------------------------

def dataset_statistics(dataset=None, num_samples=1000,
                       save_path="outputs/dataset_stats.png"):
    """Plot distributions of shapes, actions, and target positions."""
    from aiad.dataset import MixedShapeDataset
    dataset = dataset or MixedShapeDataset(num_samples=num_samples)

    tools = []
    clicks = []
    snaps = []
    ends = []
    xs = []
    ys = []

    for i in range(min(num_samples, len(dataset))):
        s = dataset[i]
        tools.append(s["target_tool"].item())
        clicks.append(s["target_click"].item())
        snaps.append(s["target_snap"].item())
        ends.append(s["target_end"].item())
        xs.append(s["target_x"].item())
        ys.append(s["target_y"].item())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Tool distribution
    ax = axes[0, 0]
    tool_counts = [tools.count(i) for i in range(NUM_TOOLS)]
    ax.barh(TOOLS, tool_counts)
    ax.set_title("Tool distribution")

    # Click/snap/end
    ax = axes[0, 1]
    labels = ["click", "snap", "end"]
    vals = [np.mean(clicks), np.mean(snaps), np.mean(ends)]
    ax.bar(labels, vals)
    ax.set_title("Action flag rates")
    ax.set_ylim(0, 1)

    # Target X histogram
    ax = axes[0, 2]
    ax.hist(xs, bins=50, alpha=0.7, color="steelblue")
    ax.set_title("Target X distribution")

    # Target Y histogram
    ax = axes[1, 0]
    ax.hist(ys, bins=50, alpha=0.7, color="coral")
    ax.set_title("Target Y distribution")

    # 2D heatmap
    ax = axes[1, 1]
    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=32)
    ax.imshow(heatmap.T, origin="lower", aspect="auto", cmap="hot")
    ax.set_title("Target position heatmap")

    # End-session distribution by tool
    ax = axes[1, 2]
    for tool_idx in range(NUM_TOOLS):
        mask = [i for i, t in enumerate(tools) if t == tool_idx]
        if mask:
            end_rate = np.mean([ends[i] for i in mask])
            ax.bar(TOOLS[tool_idx], end_rate, alpha=0.7)
    ax.set_title("End-session rate by tool")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Dataset statistics saved to {save_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="AIAD visualization toolkit")
    sub = p.add_subparsers(dest="command")

    ds = sub.add_parser("dataset", help="Visualize dataset samples")
    ds.add_argument("--num-samples", type=int, default=16)
    ds.add_argument("--save-path", default="outputs/dataset_samples.png")

    ep = sub.add_parser("episode", help="Generate episode GIF from checkpoint")
    ep.add_argument("--checkpoint", required=True)
    ep.add_argument("--save-path", default="outputs/episode.gif")

    cr = sub.add_parser("curves", help="Plot training curves")
    cr.add_argument("--checkpoint-dir", required=True)
    cr.add_argument("--save-path", default="outputs/training_curves.png")

    pr = sub.add_parser("predict", help="Visualize model predictions")
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--num-samples", type=int, default=8)
    pr.add_argument("--save-path", default="outputs/predictions.png")

    st = sub.add_parser("stats", help="Dataset statistics")
    st.add_argument("--num-samples", type=int, default=1000)
    st.add_argument("--save-path", default="outputs/dataset_stats.png")

    args = p.parse_args()

    if args.command == "dataset":
        visualize_dataset_samples(num_samples=args.num_samples, save_path=args.save_path)

    elif args.command == "episode":
        from aiad.checkpoint import load_checkpoint
        model = CADModel(in_channels=6, num_tools=NUM_TOOLS).to(DEVICE)
        load_checkpoint(args.checkpoint, model, device=DEVICE)
        model.eval()
        create_episode_gif(None, model, DEVICE, save_path=args.save_path)

    elif args.command == "curves":
        plot_training_curves(checkpoint_dir=args.checkpoint_dir, save_path=args.save_path)

    elif args.command == "predict":
        from aiad.checkpoint import load_checkpoint
        model = CADModel(in_channels=6, num_tools=NUM_TOOLS).to(DEVICE)
        load_checkpoint(args.checkpoint, model, device=DEVICE)
        model.eval()
        visualize_predictions(model, device=DEVICE, num_samples=args.num_samples,
                              save_path=args.save_path)

    elif args.command == "stats":
        dataset_statistics(num_samples=args.num_samples, save_path=args.save_path)

    else:
        p.print_help()


if __name__ == "__main__":
    main()
