"""Tests for the CAD model architecture."""

import torch
import pytest

from aiad.config import NUM_TOOLS, IMG_SIZE
from aiad.model import CADModel


@pytest.fixture
def model():
    return CADModel(in_channels=6, base_channels=16, num_tools=NUM_TOOLS)


@pytest.fixture
def dummy_input():
    B = 2
    obs = torch.randn(B, 6, IMG_SIZE, IMG_SIZE)
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
        assert out["x"].shape == (2, IMG_SIZE)

    def test_y_shape(self, model, dummy_input):
        out = model(*dummy_input)
        assert out["y"].shape == (2, IMG_SIZE)

    def test_tool_shape(self, model, dummy_input):
        out = model(*dummy_input)
        assert out["tool"].shape == (2, NUM_TOOLS)

    def test_binary_head_shapes(self, model, dummy_input):
        out = model(*dummy_input)
        for key in ("click", "snap", "end", "value"):
            assert out[key].shape == (2, 1)


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
