"""
PPO fine-tuning: the model draws entire images in the environment and is
rewarded based on thick-line IoU between its drawing and the target.

Usage:
    python -m aiad.train_ppo --pretrained checkpoints/supervised/best.pt [OPTIONS]

Or from a notebook:
    from aiad.train_ppo import train
    train(pretrained="checkpoints/supervised/best.pt", num_episodes=500)
"""

import argparse
import os
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from aiad.checkpoint import BestModelTracker, load_checkpoint
from aiad.config import DEVICE, IMG_SIZE, NUM_TOOLS
from aiad.env import MixedCADEnvironment
from aiad.model import CADModel
from aiad.viz import create_episode_gif


def _compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def _collect_episode(env, model, device):
    """Roll out one episode, storing transitions for PPO."""
    obs_tuple = env.reset()
    transitions = []
    done = False
    ep_reward = 0.0

    while not done:
        obs, pt, pc = obs_tuple
        obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
        pt_t = torch.tensor([pt], dtype=torch.long, device=device)
        pc_t = torch.tensor([[pc]], dtype=torch.float32, device=device)

        with torch.no_grad():
            action, log_prob, value, _ = model.get_action(obs_t, pt_t, pc_t)

        env_action = {k: v.item() for k, v in action.items()}
        next_obs_tuple, reward, done, info = env.step(env_action)

        transitions.append({
            "obs": obs, "prev_tool": pt, "prev_click": pc,
            "action": {k: v.item() for k, v in action.items()},
            "log_prob": log_prob.item(),
            "value": value.item(),
            "reward": reward,
            "done": float(done),
        })
        obs_tuple = next_obs_tuple
        ep_reward += reward

    return transitions, ep_reward, info


def _ppo_update(model, optimizer, transitions, device,
                clip_ratio=0.2, value_coeff=0.5, entropy_coeff=0.01,
                max_grad_norm=1.0, ppo_epochs=4, gamma=0.99, lam=0.95):
    """Run multiple PPO epochs on one episode's transitions."""
    T = len(transitions)

    # Tensorize
    obs_batch = torch.from_numpy(
        np.stack([t["obs"] for t in transitions])
    ).float().to(device)
    pt_batch = torch.tensor(
        [t["prev_tool"] for t in transitions], dtype=torch.long, device=device
    )
    pc_batch = torch.tensor(
        [[t["prev_click"]] for t in transitions], dtype=torch.float32, device=device
    )
    old_log_probs = torch.tensor(
        [t["log_prob"] for t in transitions], dtype=torch.float32, device=device
    )
    rewards = torch.tensor([t["reward"] for t in transitions], dtype=torch.float32)
    values = torch.tensor([t["value"] for t in transitions], dtype=torch.float32)
    dones = torch.tensor([t["done"] for t in transitions], dtype=torch.float32)

    action_batch = {
        k: torch.tensor([t["action"][k] for t in transitions],
                        dtype=torch.long if k in ("x", "y", "tool") else torch.float32,
                        device=device)
        for k in ("x", "y", "tool", "click", "snap", "end")
    }

    advantages, returns = _compute_gae(rewards, values, dones, gamma, lam)
    advantages = advantages.to(device)
    returns = returns.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_loss_acc = 0.0
    for _ in range(ppo_epochs):
        new_log_probs, new_values, entropy = model.evaluate_action(
            obs_batch, pt_batch, pc_batch, action_batch
        )

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, returns)
        entropy_loss = -entropy.mean()

        loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss_acc += loss.item()

    return total_loss_acc / ppo_epochs


def train(
    num_episodes=1500,
    lr=1e-5,
    gamma=0.99,
    clip_ratio=0.2,
    value_coeff=0.5,
    entropy_coeff=0.01,
    max_grad_norm=1.0,
    ppo_epochs=4,
    lam=0.95,
    max_steps=30,
    reward_thickness=10,
    checkpoint_dir="checkpoints/ppo",
    output_dir="outputs/ppo",
    resume=None,
    pretrained=None,
    viz_interval=50,
    log_interval=10,
    device=None,
):
    """Run PPO training. Callable from scripts or notebooks."""
    device = device or DEVICE
    os.makedirs(output_dir, exist_ok=True)

    model = CADModel(in_channels=6, num_tools=NUM_TOOLS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start_ep = 0

    if resume:
        meta = load_checkpoint(resume, model, optimizer, device)
        start_ep = meta.get("episode", 0)
        print(f"Resumed from {resume} (episode {start_ep})")
    elif pretrained:
        load_checkpoint(pretrained, model, device=device)
        print(f"Loaded pre-trained weights from {pretrained}")

    tracker = BestModelTracker(checkpoint_dir, metric_name="avg_iou", mode="max")
    env = MixedCADEnvironment(
        img_size=IMG_SIZE, max_steps=max_steps, reward_thickness=reward_thickness
    )

    # Graceful interrupt
    interrupted = False
    def _handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nInterrupted — saving checkpoint …")
    signal.signal(signal.SIGINT, _handler)

    reward_history = []
    iou_history = []
    pbar = tqdm(range(start_ep, start_ep + num_episodes), desc="PPO")

    for ep in pbar:
        if interrupted:
            break

        model.train()
        transitions, ep_reward, info = _collect_episode(env, model, device)
        loss = _ppo_update(
            model, optimizer, transitions, device,
            clip_ratio=clip_ratio, value_coeff=value_coeff,
            entropy_coeff=entropy_coeff, max_grad_norm=max_grad_norm,
            ppo_epochs=ppo_epochs, gamma=gamma, lam=lam,
        )

        ep_iou = info.get("iou", 0.0)
        reward_history.append(ep_reward)
        iou_history.append(ep_iou)

        if (ep + 1) % log_interval == 0:
            recent_r = np.mean(reward_history[-log_interval:])
            recent_iou = np.mean(iou_history[-log_interval:])
            pbar.set_postfix({
                "R": f"{recent_r:.1f}",
                "IoU": f"{recent_iou:.3f}",
                "loss": f"{loss:.3f}",
            })

        if (ep + 1) % viz_interval == 0:
            model.eval()
            gif_path = f"{output_dir}/episode_{ep + 1}.gif"
            create_episode_gif(env, model, device, save_path=gif_path)
            recent_iou = np.mean(iou_history[-viz_interval:])
            is_best = tracker.update(
                model, optimizer, recent_iou,
                metadata={"episode": ep + 1, "avg_reward": np.mean(reward_history[-viz_interval:])},
            )
            if is_best:
                tqdm.write(f"  New best avg IoU: {recent_iou:.4f}")

    # Final save
    if iou_history:
        final_iou = np.mean(iou_history[-min(50, len(iou_history)):])
        tracker.update(model, optimizer, final_iou,
                       metadata={"episode": ep + 1, "final": True})
    print(f"PPO training complete. Checkpoints in {checkpoint_dir}/")
    return model


def main():
    p = argparse.ArgumentParser(description="PPO fine-tuning for CAD tracer")
    p.add_argument("--num-episodes", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--reward-thickness", type=int, default=10)
    p.add_argument("--checkpoint-dir", default="checkpoints/ppo")
    p.add_argument("--output-dir", default="outputs/ppo")
    p.add_argument("--resume", default=None, help="Resume PPO from checkpoint")
    p.add_argument("--pretrained", default=None, help="Load supervised pre-trained weights")
    p.add_argument("--viz-interval", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)
    args = p.parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()
