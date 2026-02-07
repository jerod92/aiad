"""Tests for the CAD model architecture."""

import torch
import pytest

from aiad.config import NUM_TOOLS
from aiad.model import CADModel


@pytest.fixture
def model():
    return CADModel(in_channels=6, base_channels=16, num_tools=NUM_TOOLS,
                    transformer_layers=1, transformer_heads=2)


@pytest.fixture
def dummy_input():
    B = 2
    S = 256
    obs = torch.randn(B, 6, S, S)
    prev_tool = torch.zeros(B, dtype=torch.long)
    prev_click = torch.zeros(B, 1)
    return obs, prev_tool, prev_click


class TestForward:
    def test_output_keys(self, model, dummy_input):
        out = model(*dummy_input)
        for key in ("x", "y", "tool", "click", "snap", "end", "value"):
            assert key in out

    def test_x_shape(self, model, dummy_input):
        out = model(*dummy_input)
        assert out["x"].shape == (2, 256)

    def test_y_shape(self, model, dummy_input):
        out = model(*dummy_input)
        assert out["y"].shape == (2, 256)

    def test_tool_shape(self, model, dummy_input):
        out = model(*dummy_input)
        assert out["tool"].shape == (2, NUM_TOOLS)

    def test_binary_head_shapes(self, model, dummy_input):
        out = model(*dummy_input)
        for key in ("click", "snap", "end", "value"):
            assert out[key].shape == (2, 1)


class TestFromPreset:
    def test_mini_preset(self):
        model = CADModel.from_preset("mini")
        assert model is not None
        obs = torch.randn(1, 6, 256, 256)
        pt = torch.zeros(1, dtype=torch.long)
        pc = torch.zeros(1, 1)
        out = model(obs, pt, pc)
        assert out["tool"].shape == (1, NUM_TOOLS)

    def test_large_preset(self):
        model = CADModel.from_preset("large")
        assert model is not None


class TestGetAction:
    def test_deterministic(self, model, dummy_input):
        action, log_prob, value, entropy = model.get_action(*dummy_input, deterministic=True)
        assert action["x"].shape == (2,)
        assert action["y"].shape == (2,)
        assert log_prob.shape == (2,)

    def test_stochastic(self, model, dummy_input):
        action, log_prob, value, entropy = model.get_action(*dummy_input, deterministic=False)
        assert action["x"].shape == (2,)
        assert log_prob.shape == (2,)
        assert entropy.shape == (2,)
        assert (entropy > 0).all()


class TestEvaluateAction:
    def test_shapes(self, model, dummy_input):
        action, old_lp, _, _ = model.get_action(*dummy_input, deterministic=False)
        lp, val, ent = model.evaluate_action(*dummy_input, action)
        assert lp.shape == (2,)
        assert val.shape == (2,)
        assert ent.shape == (2,)


class TestGradients:
    def test_gradients_flow(self, model, dummy_input):
        out = model(*dummy_input)
        loss = out["x"].sum() + out["y"].sum()
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0
        assert any(g > 0 for g in grad_norms)
