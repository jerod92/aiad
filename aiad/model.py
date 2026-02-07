"""
CAD tracing model: UNet encoder -> Transformer bridge -> UNet decoder.

Supports two presets:
  * **large** — 512 px, base_channels=32, 2-layer transformer  (serious training)
  * **mini**  — 256 px, base_channels=16, 1-layer transformer  (Kaggle / PoC)

The bridge injects tool/click state before transformer processing.
The decoder produces factored X/Y position distributions via marginal pooling.
Auxiliary heads predict tool, click, snap, end-session, and value (for PPO).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

from aiad.config import NUM_TOOLS, MODEL_PRESETS, PRESET_LARGE


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _conv_block(in_ch, out_ch):
    gn_groups = min(32, out_ch)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(gn_groups, out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(gn_groups, out_ch),
        nn.SiLU(),
    )


class UNetEncoder(nn.Module):
    """Four-stage UNet encoder with max-pooling between stages."""

    def __init__(self, in_channels=6, base_channels=32):
        super().__init__()
        bc = base_channels
        self.enc1 = _conv_block(in_channels, bc)
        self.enc2 = _conv_block(bc, bc * 2)
        self.enc3 = _conv_block(bc * 2, bc * 4)
        self.enc4 = _conv_block(bc * 4, bc * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        bottleneck = self.pool(s4)
        return bottleneck, [s1, s2, s3, s4]


class TransformerBridge(nn.Module):
    """Transformer bridge that injects tool/click state before processing."""

    def __init__(self, channels=256, num_heads=4, num_layers=2, num_tools=NUM_TOOLS):
        super().__init__()
        self.tool_embed = nn.Embedding(num_tools, channels)
        self.click_embed = nn.Linear(1, channels)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=num_heads,
                dim_feedforward=channels * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, prev_tool, prev_click):
        B, C, H, W = x.shape
        state = self.tool_embed(prev_tool) + self.click_embed(prev_click)
        x = x + state.view(B, C, 1, 1)
        x = self.transformer(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).reshape(B, C, H, W)


class UNetDecoder(nn.Module):
    """Four-stage UNet decoder with skip connections and marginal X/Y heads."""

    def __init__(self, base_channels=32):
        super().__init__()
        bc = base_channels
        self.up1 = nn.ConvTranspose2d(bc * 8, bc * 8, 2, stride=2)
        self.dec1 = _conv_block(bc * 16, bc * 4)
        self.up2 = nn.ConvTranspose2d(bc * 4, bc * 4, 2, stride=2)
        self.dec2 = _conv_block(bc * 8, bc * 2)
        self.up3 = nn.ConvTranspose2d(bc * 2, bc * 2, 2, stride=2)
        self.dec3 = _conv_block(bc * 4, bc)
        self.up4 = nn.ConvTranspose2d(bc, bc, 2, stride=2)
        self.dec4 = _conv_block(bc * 2, bc)
        # Marginal heads: project to 1-channel map then collapse one spatial dim
        self.x_head = nn.Conv2d(bc, 1, 1)
        self.y_head = nn.Conv2d(bc, 1, 1)

    def forward(self, x, skips):
        s1, s2, s3, s4 = skips
        x = self.dec1(torch.cat([self.up1(x), s4], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s3], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s2], dim=1))
        x = self.dec4(torch.cat([self.up4(x), s1], dim=1))
        # x_logits: mean over height -> [B, W]  (column scores)
        x_logits = self.x_head(x).mean(dim=2).squeeze(1)
        # y_logits: mean over width  -> [B, H]  (row scores)
        y_logits = self.y_head(x).mean(dim=3).squeeze(1)
        return x_logits, y_logits


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CADModel(nn.Module):
    """Complete CAD-tracing model.

    Inputs
    ------
    obs : [B, 6, H, W]
        Channels: target_R, target_G, target_B, drawing, ghost, cursor.
    prev_tool : [B]  (long)
    prev_click : [B, 1]  (float)

    Outputs (dict)
    -------
    x, y       : logits [B, W] / [B, H] for cursor position
    tool       : logits [B, NUM_TOOLS]   (24 slots)
    click      : logit  [B, 1]
    snap       : logit  [B, 1]
    end        : logit  [B, 1]
    value      : scalar [B, 1]  (state-value for PPO)
    """

    def __init__(self, in_channels=6, base_channels=32, num_tools=NUM_TOOLS,
                 transformer_layers=2, transformer_heads=4):
        super().__init__()
        bridge_dim = base_channels * 8
        self.encoder = UNetEncoder(in_channels, base_channels)
        self.bridge = TransformerBridge(
            channels=bridge_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            num_tools=num_tools,
        )
        self.decoder = UNetDecoder(base_channels)
        self.aux_pool = nn.AdaptiveAvgPool2d(1)
        self.tool_head = nn.Sequential(
            nn.Linear(bridge_dim, 128), nn.SiLU(), nn.Linear(128, num_tools),
        )
        self.click_head = nn.Sequential(
            nn.Linear(bridge_dim, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        self.snap_head = nn.Sequential(
            nn.Linear(bridge_dim, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        self.end_head = nn.Sequential(
            nn.Linear(bridge_dim, 64), nn.SiLU(), nn.Linear(64, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(bridge_dim, 64), nn.SiLU(), nn.Linear(64, 1),
        )

    @classmethod
    def from_preset(cls, preset_name="large", **overrides):
        """Construct a model from a named preset ('large' or 'mini')."""
        preset = MODEL_PRESETS[preset_name]
        kwargs = dict(
            in_channels=6,
            base_channels=preset.base_channels,
            num_tools=NUM_TOOLS,
            transformer_layers=preset.transformer_layers,
            transformer_heads=preset.transformer_heads,
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def forward(self, obs, prev_tool, prev_click):
        encoded, skips = self.encoder(obs)
        bridge_out = self.bridge(encoded, prev_tool, prev_click)
        x_logits, y_logits = self.decoder(bridge_out, skips)
        flat = self.aux_pool(bridge_out).flatten(1)
        return {
            "x": x_logits,
            "y": y_logits,
            "tool": self.tool_head(flat),
            "click": self.click_head(flat),
            "snap": self.snap_head(flat),
            "end": self.end_head(flat),
            "value": self.value_head(flat),
        }

    def get_action(self, obs, prev_tool, prev_click, deterministic=False):
        """Sample (or greedily pick) an action; return action dict, log_prob, value."""
        out = self.forward(obs, prev_tool, prev_click)
        if deterministic:
            action = {
                "x": out["x"].argmax(dim=1),
                "y": out["y"].argmax(dim=1),
                "tool": out["tool"].argmax(dim=1),
                "click": (torch.sigmoid(out["click"]) > 0.5).float(),
                "snap": (torch.sigmoid(out["snap"]) > 0.5).float(),
                "end": (torch.sigmoid(out["end"]) > 0.5).float(),
            }
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
            entropy = torch.zeros(obs.shape[0], device=obs.device)
        else:
            x_dist = Categorical(logits=out["x"])
            y_dist = Categorical(logits=out["y"])
            tool_dist = Categorical(logits=out["tool"])
            click_dist = Bernoulli(logits=out["click"].squeeze(-1))
            snap_dist = Bernoulli(logits=out["snap"].squeeze(-1))
            end_dist = Bernoulli(logits=out["end"].squeeze(-1))

            action = {
                "x": x_dist.sample(),
                "y": y_dist.sample(),
                "tool": tool_dist.sample(),
                "click": click_dist.sample(),
                "snap": snap_dist.sample(),
                "end": end_dist.sample(),
            }
            log_prob = (
                x_dist.log_prob(action["x"])
                + y_dist.log_prob(action["y"])
                + tool_dist.log_prob(action["tool"])
                + click_dist.log_prob(action["click"])
                + snap_dist.log_prob(action["snap"])
                + end_dist.log_prob(action["end"])
            )
            entropy = (
                x_dist.entropy()
                + y_dist.entropy()
                + tool_dist.entropy()
                + click_dist.entropy()
                + snap_dist.entropy()
                + end_dist.entropy()
            )

        return action, log_prob, out["value"].squeeze(-1), entropy

    def evaluate_action(self, obs, prev_tool, prev_click, action):
        """Re-evaluate a stored action: return log_prob, value, entropy."""
        out = self.forward(obs, prev_tool, prev_click)
        x_dist = Categorical(logits=out["x"])
        y_dist = Categorical(logits=out["y"])
        tool_dist = Categorical(logits=out["tool"])
        click_dist = Bernoulli(logits=out["click"].squeeze(-1))
        snap_dist = Bernoulli(logits=out["snap"].squeeze(-1))
        end_dist = Bernoulli(logits=out["end"].squeeze(-1))

        log_prob = (
            x_dist.log_prob(action["x"])
            + y_dist.log_prob(action["y"])
            + tool_dist.log_prob(action["tool"])
            + click_dist.log_prob(action["click"])
            + snap_dist.log_prob(action["snap"])
            + end_dist.log_prob(action["end"])
        )
        entropy = (
            x_dist.entropy()
            + y_dist.entropy()
            + tool_dist.entropy()
            + click_dist.entropy()
            + snap_dist.entropy()
            + end_dist.entropy()
        )
        return log_prob, out["value"].squeeze(-1), entropy
