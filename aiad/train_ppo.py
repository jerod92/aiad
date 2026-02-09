"""
PPO fine-tuning: the model draws entire images in the environment and is
rewarded based on thick-line IoU between its drawing and the target.

Supports ``--model-size large|mini`` to select the model preset.

Usage:
    python -m aiad.train_ppo --pretrained checkpoints/supervised/best.pt [OPTIONS]

Or from a notebook:
    from aiad.train_ppo import train
    train(pretrained="checkpoints/supervised/best.pt", num_episodes=500, model_size="mini")
"""

import argparse
import os
import signal
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from aiad.checkpoint import BestModelTracker, load_checkpoint
from aiad.config import DEVICE, NUM_TOOLS, MODEL_PRESETS
from aiad.env import MixedCADEnvironment
from aiad.model import CADModel
from aiad.viz import create_episode_gif


def _compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation with proper bootstrapping for timeouts.
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0
    
    # We append the 'next_value' to the end of values to simplify logic
    # values now has shape (T + 1)
    values = torch.cat([values, torch.tensor([next_value], device=values.device)])

    for t in reversed(range(T)):
        # If done[t] is 1 (True), we disregard the future value.
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        
    returns = advantages + values[:-1]
    return advantages, returns


def _collect_episode(env, model, device):
    """
    Roll out one episode. 
    Returns: transitions list, total_reward, info, and last_obs_tuple (for bootstrapping)
    """
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

    # Calculate Value of the final state (Bootstrapping)
    # If the episode ended via timeout (max_steps), we need V(s_last).
    # If it ended via termination (success/fail), V(s_last) is effectively 0, 
    # but we handle that via the mask in GAE.
    obs, pt, pc = obs_tuple
    obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
    pt_t = torch.tensor([pt], dtype=torch.long, device=device)
    pc_t = torch.tensor([[pc]], dtype=torch.float32, device=device)
    with torch.no_grad():
        _, _, next_val, _ = model.get_action(obs_t, pt_t, pc_t)
        
    return transitions, ep_reward, info, next_val.item()


def _ppo_update(model, optimizer, buffer, device,
                clip_ratio=0.2, value_coeff=0.5, entropy_coeff=0.01,
                max_grad_norm=0.5, ppo_epochs=4, batch_size=64):
    """
    Run PPO epochs on a collected buffer using Mini-batches.
    """
    # 1. Flatten buffer into tensors
    obs_batch = torch.from_numpy(np.stack([t["obs"] for t in buffer])).float().to(device)
    pt_batch = torch.tensor([t["prev_tool"] for t in buffer], dtype=torch.long, device=device)
    pc_batch = torch.tensor([[t["prev_click"]] for t in buffer], dtype=torch.float32, device=device)
    
    old_log_probs = torch.tensor([t["log_prob"] for t in buffer], dtype=torch.float32, device=device)
    returns = torch.tensor([t["return"] for t in buffer], dtype=torch.float32, device=device)
    advantages = torch.tensor([t["advantage"] for t in buffer], dtype=torch.float32, device=device)
    
    # Normalize advantages (Critical for stability)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    action_batch = {
        k: torch.tensor([t["action"][k] for t in buffer],
                        dtype=torch.long if k in ("x", "y", "tool") else torch.float32,
                        device=device)
        for k in ("x", "y", "tool", "click", "snap", "end")
    }

    dataset_size = len(buffer)
    indices = np.arange(dataset_size)
    total_loss_acc = 0.0
    updates = 0

    # 2. Mini-batch Training
    for _ in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            # Slice batches
            mb_obs = obs_batch[idx]
            mb_pt = pt_batch[idx]
            mb_pc = pc_batch[idx]
            mb_actions = {k: v[idx] for k, v in action_batch.items()}
            mb_old_log_probs = old_log_probs[idx]
            mb_adv = advantages[idx]
            mb_ret = returns[idx]

            new_log_probs, new_values, entropy = model.evaluate_action(
                mb_obs, mb_pt, mb_pc, mb_actions
            )

            # PPO Logic
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(new_values, mb_ret)
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            total_loss_acc += loss.item()
            updates += 1

    return total_loss_acc / updates


def train(
    num_episodes=2000, # Increased default, PPO needs time
    lr=1e-5,
    gamma=0.99,
    clip_ratio=0.2,
    value_coeff=0.5,
    entropy_coeff=0.01,
    max_grad_norm=0.5, # Tighter clipping
    ppo_epochs=4,
    lam=0.95,
    max_steps=50,
    reward_thickness=10,
    model_size="large",
    checkpoint_dir="checkpoints/ppo",
    output_dir="outputs/ppo",
    resume=None,
    pretrained=None,
    viz_interval=50,
    log_interval=10,
    update_timesteps=2048, # New: How many steps to collect before update
    batch_size=64,         # New: Mini-batch size
    device=None,
):
    device = device or DEVICE
    os.makedirs(output_dir, exist_ok=True)

    preset = MODEL_PRESETS[model_size]
    img_size = preset.img_size

    model = CADModel.from_preset(model_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Initialize trackers
    start_ep = 0
    if resume:
        meta = load_checkpoint(resume, model, optimizer, device)
        start_ep = meta.get("episode", 0)
    elif pretrained:
        load_checkpoint(pretrained, model, device=device)

    tracker = BestModelTracker(checkpoint_dir, metric_name="avg_iou", mode="max")
    env = MixedCADEnvironment(
        img_size=img_size, max_steps=max_steps, reward_thickness=reward_thickness
    )

    # Buffer for PPO
    buffer = []
    buffer_steps = 0
    
    reward_history = []
    iou_history = []
    
    # Interruption handling
    interrupted = False
    def _handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\nInterrupted — saving checkpoint ...")
    signal.signal(signal.SIGINT, _handler)

    pbar = tqdm(range(start_ep, start_ep + num_episodes), desc="PPO")

    for ep in pbar:
        if interrupted: break

        model.eval() # Collect in eval mode usually (or train with no_grad), but for BN/Dropout often keep train()
        # However, for PPO collection, we usually want exploration but consistent stats. 
        # Let's keep model.train() to match original behavior but handle no_grad inside collect
        model.train() 
        
        # 1. Collect Data
        transitions, ep_reward, info, next_val = _collect_episode(env, model, device)
        
        # 2. Compute GAE for this episode
        rewards = torch.tensor([t["reward"] for t in transitions], dtype=torch.float32)
        values = torch.tensor([t["value"] for t in transitions], dtype=torch.float32)
        dones = torch.tensor([t["done"] for t in transitions], dtype=torch.float32)
        
        advantages, returns = _compute_gae(rewards, values, dones, next_val, gamma, lam)
        
        # Store processed data in buffer
        for i, t in enumerate(transitions):
            t["advantage"] = advantages[i].item()
            t["return"] = returns[i].item()
            buffer.append(t)
            
        buffer_steps += len(transitions)
        
        # Logging
        ep_iou = info.get("iou", 0.0)
        reward_history.append(ep_reward)
        iou_history.append(ep_iou)
        
        loss_disp = "wait"

        # 3. Update Policy only when buffer is full
        if buffer_steps >= update_timesteps:
            loss = _ppo_update(
                model, optimizer, buffer, device,
                clip_ratio=clip_ratio, value_coeff=value_coeff,
                entropy_coeff=entropy_coeff, max_grad_norm=max_grad_norm,
                ppo_epochs=ppo_epochs, batch_size=batch_size,
            )
            loss_disp = f"{loss:.3f}"
            buffer = [] # Clear buffer
            buffer_steps = 0

        if (ep + 1) % log_interval == 0:
            recent_r = np.mean(reward_history[-log_interval:])
            recent_iou = np.mean(iou_history[-log_interval:])
            pbar.set_postfix({
                "R": f"{recent_r:.1f}",
                "IoU": f"{recent_iou:.3f}",
                "loss": loss_disp,
            })

        if (ep + 1) % viz_interval == 0:
            model.eval()
            create_episode_gif(env, model, device, save_path=f"{output_dir}/episode_{ep + 1}.gif")
            recent_iou = np.mean(iou_history[-viz_interval:])
            tracker.update(model, optimizer, recent_iou, {"episode": ep + 1, "model_size": model_size})
            model.train()

    if iou_history:
        final_iou = np.mean(iou_history[-min(50, len(iou_history)):])
        tracker.update(model, optimizer, final_iou, {"episode": ep + 1, "model_size": model_size, "final": True})
    
    return model

def main():
    p = argparse.ArgumentParser()
    # ... (Same args as before)
    p.add_argument("--num-episodes", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5) # Reduced default
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--reward-thickness", type=int, default=10)
    p.add_argument("--model-size", choices=["large", "mini"], default="large")
    p.add_argument("--checkpoint-dir", default="checkpoints/ppo")
    p.add_argument("--output-dir", default="outputs/ppo")
    p.add_argument("--resume", default=None)
    p.add_argument("--pretrained", default=None)
    p.add_argument("--viz-interval", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=10)
    # New args
    p.add_argument("--update-timesteps", type=int, default=2000, help="Timesteps per PPO update")
    p.add_argument("--batch-size", type=int, default=64, help="PPO mini-batch size")
    
    args = p.parse_args()
    train(**vars(args))

if __name__ == "__main__":
    main()
