"""Tests for the synthetic dataset."""

import torch
import pytest

from aiad.config import IMG_SIZE, NUM_TOOLS
from aiad.dataset import MixedShapeDataset


@pytest.fixture
def dataset():
    return MixedShapeDataset(num_samples=50, img_size=IMG_SIZE)


class TestBasics:
    def test_length(self, dataset):
        assert len(dataset) == 50

    def test_getitem_returns_dict(self, dataset):
        s = dataset[0]
        assert isinstance(s, dict)

    def test_required_keys(self, dataset):
        s = dataset[0]
        for key in ("obs", "prev_tool", "prev_click",
                     "target_x", "target_y", "target_tool",
                     "target_click", "target_snap", "target_end"):
            assert key in s


class TestShapes:
    def test_obs_shape(self, dataset):
        s = dataset[0]
        assert s["obs"].shape == (6, IMG_SIZE, IMG_SIZE)

    def test_obs_dtype(self, dataset):
        s = dataset[0]
        assert s["obs"].dtype == torch.float32


class TestValueRanges:
    def test_obs_in_0_1(self, dataset):
        for _ in range(10):
            s = dataset[0]
            assert s["obs"].min() >= 0.0
            assert s["obs"].max() <= 1.0

    def test_target_x_in_range(self, dataset):
        for _ in range(10):
            s = dataset[0]
            assert 0 <= s["target_x"].item() < IMG_SIZE

    def test_target_y_in_range(self, dataset):
        for _ in range(10):
            s = dataset[0]
            assert 0 <= s["target_y"].item() < IMG_SIZE

    def test_target_tool_valid(self, dataset):
        for _ in range(10):
            s = dataset[0]
            assert 0 <= s["target_tool"].item() < NUM_TOOLS

    def test_binary_flags(self, dataset):
        for _ in range(10):
            s = dataset[0]
            for key in ("target_click", "target_snap", "target_end"):
                assert s[key].item() in (0.0, 1.0)


class TestDiversity:
    def test_multiple_tools_appear(self):
        ds = MixedShapeDataset(num_samples=200)
        tools = set()
        for i in range(200):
            tools.add(ds[i]["target_tool"].item())
        assert len(tools) >= 2, f"Only saw tools: {tools}"
