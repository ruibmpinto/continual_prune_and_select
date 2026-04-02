"""Tests for task_selector module.

Tests
-----
test_register_and_select
    Register tasks and verify selection works.
test_single_task_returns_zero
    With one task, always returns task 0.
test_save_load
    Database persists through save/load cycle.
"""

#
#                                                                Modules
# =====================================================================
# Standard
import os
import tempfile

# Third-party
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# Local
from fingerprinting.task_selector import (
    FingerprintTaskSelector,
)

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def _make_dummy_model():
    """Create a small CNN with layer2 and layer3."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.task_id = 0
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.layer2 = nn.Sequential(
                nn.Conv2d(8, 16, 3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return DummyModel()


def _dummy_set_task(model, task_id):
    """Minimal set_task for dummy model."""
    model.task_id = task_id


def _make_task_dataset(n_samples=30, seed=0, scale=1.0):
    """Create a synthetic dataset for one task."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(
        n_samples, 3, 32, 32, generator=rng,
    ) * scale
    y = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(x, y)


# =====================================================================
class TestFingerprintTaskSelector:
    """Tests for FingerprintTaskSelector."""

    # -----------------------------------------------------------------
    def test_register_and_select(self):
        """Register tasks and verify selection works."""
        model = _make_dummy_model()
        model.eval()
        device = torch.device('cpu')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        selector = FingerprintTaskSelector(
            model, device, _dummy_set_task,
            num_register_samples=20,
        )
        # Register 3 tasks with different data
        datasets = {}
        for i in range(3):
            datasets[i] = _make_task_dataset(
                n_samples=20, seed=i * 1000,
                scale=float(i + 1),
            )
            selector.register_task(i, datasets[i])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Query with data from task 0
        x_query = datasets[0].tensors[0][:5]
        task_id = selector.select_task(x_query, 3)
        # Should return a valid task ID
        assert 0 <= task_id < 3

    # -----------------------------------------------------------------
    def test_single_task_returns_zero(self):
        """With one task, always returns task 0."""
        model = _make_dummy_model()
        model.eval()
        device = torch.device('cpu')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        selector = FingerprintTaskSelector(
            model, device, _dummy_set_task,
            num_register_samples=10,
        )
        dataset = _make_task_dataset(n_samples=10)
        selector.register_task(0, dataset)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        x = torch.randn(4, 3, 32, 32)
        task_id = selector.select_task(x, 1)
        assert task_id == 0

    # -----------------------------------------------------------------
    def test_save_load(self):
        """Database persists through save/load cycle."""
        model = _make_dummy_model()
        model.eval()
        device = torch.device('cpu')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        selector = FingerprintTaskSelector(
            model, device, _dummy_set_task,
            num_register_samples=10,
        )
        dataset = _make_task_dataset(n_samples=10)
        selector.register_task(0, dataset)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        with tempfile.NamedTemporaryFile(
            suffix='.pt', delete=False,
        ) as f:
            path = f.name
        try:
            selector.save(path)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            selector2 = FingerprintTaskSelector(
                model, device, _dummy_set_task,
            )
            selector2.load(path)
            assert selector2._db.num_tasks == 1
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            x = torch.randn(4, 3, 32, 32)
            task_id = selector2.select_task(x, 1)
            assert task_id == 0
        finally:
            os.unlink(path)
