"""Checkpoint save / load / best-model tracking."""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types (bool_, int64, float64, etc.)."""

    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_checkpoint(model, optimizer, metadata, path):
    """Persist model + optimizer state and arbitrary metadata."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device=None):
    """Restore model (and optionally optimizer) from *path*. Returns metadata."""
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("metadata", {})


class BestModelTracker:
    """Save *latest* every call; save *best* when the tracked metric improves.

    Also appends every update to a JSON history file for plotting curves.
    """

    def __init__(self, checkpoint_dir="checkpoints", metric_name="loss", mode="min"):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.history_file = self.dir / "history.json"
        self.history: list[dict] = []
        # Load existing history if resuming
        if self.history_file.exists():
            with open(self.history_file) as f:
                self.history = json.load(f)
            if self.history:
                vals = [h[self.metric_name] for h in self.history if self.metric_name in h]
                if vals:
                    self.best_value = min(vals) if mode == "min" else max(vals)

    def update(self, model, optimizer, metric_value, metadata=None):
        """Save latest; conditionally save best. Returns True if new best."""
        metadata = metadata or {}
        metadata[self.metric_name] = metric_value

        save_checkpoint(model, optimizer, metadata, str(self.dir / "latest.pt"))

        is_best = (
            (self.mode == "min" and metric_value < self.best_value)
            or (self.mode == "max" and metric_value > self.best_value)
        )
        if is_best:
            self.best_value = float(metric_value)
            save_checkpoint(model, optimizer, metadata, str(self.dir / "best.pt"))

        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "is_best": bool(is_best),
            **metadata,
        })
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2, cls=_NumpyEncoder)

        return is_best
