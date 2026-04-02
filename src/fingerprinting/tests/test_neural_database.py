"""Tests for neural_database module.

Tests
-----
test_register_task
    DB stores task with non-empty hashes.
test_num_tasks
    Property tracks registered tasks.
test_match_correct_task
    Query from task's own data matches that task.
test_single_task
    With one task, always returns task 0.
test_empty_query
    Empty query returns task 0 with zero scores.
"""

#
#                                                                Modules
# =====================================================================
# Standard
from collections import Counter

# Third-party
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# Local
from fingerprinting.neural_database import (
    TaskFingerprintDatabase,
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


def _make_task_dataset(n_samples=50, seed=0):
    """Create a synthetic dataset for one task."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, 3, 32, 32,
                     generator=rng)
    y = torch.zeros(n_samples, dtype=torch.long)
    return TensorDataset(x, y)


# =====================================================================
class TestTaskFingerprintDatabase:
    """Tests for TaskFingerprintDatabase."""

    # -----------------------------------------------------------------
    def test_register_task(self):
        """DB stores task with non-empty hashes."""
        model = _make_dummy_model()
        device = torch.device('cpu')
        dataset = _make_task_dataset(n_samples=20)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        db = TaskFingerprintDatabase()
        db.register_task(
            0, model, dataset, device,
            num_samples=20, batch_size=10,
        )
        assert db.num_tasks == 1
        assert len(db._task_hashes[0]) > 0

    # -----------------------------------------------------------------
    def test_num_tasks(self):
        """Property tracks registered tasks."""
        model = _make_dummy_model()
        device = torch.device('cpu')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        db = TaskFingerprintDatabase()
        for i in range(3):
            dataset = _make_task_dataset(
                n_samples=10, seed=i,
            )
            db.register_task(
                i, model, dataset, device,
                num_samples=10,
            )
        assert db.num_tasks == 3

    # -----------------------------------------------------------------
    def test_match_correct_task(self):
        """Query from task data matches that task.

        Uses different random seeds per task to create
        distinguishable activation distributions.
        """
        model = _make_dummy_model()
        model.eval()
        device = torch.device('cpu')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Register 3 tasks with different data distributions
        db = TaskFingerprintDatabase()
        datasets = {}
        for i in range(3):
            # Use very different data per task
            rng = torch.Generator().manual_seed(i * 1000)
            x = torch.randn(30, 3, 32, 32,
                             generator=rng) * (i + 1)
            y = torch.zeros(30, dtype=torch.long)
            datasets[i] = TensorDataset(x, y)
            db.register_task(
                i, model, datasets[i], device,
                num_samples=30,
            )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Query with data from each task
        from fingerprinting.neural_fingerprint import (
            fingerprint_batch,
        )
        for task_id in range(3):
            x_query = datasets[task_id].tensors[0][:10]
            query_hashes = fingerprint_batch(
                model, x_query, device,
            )
            best, scores = db.match(query_hashes, 3)
            # The correct task should have highest score
            assert scores[task_id] == max(scores), (
                f'Task {task_id}: expected highest score '
                f'{scores[task_id]}, got scores {scores}'
            )

    # -----------------------------------------------------------------
    def test_single_task(self):
        """With one task, always returns task 0."""
        model = _make_dummy_model()
        device = torch.device('cpu')
        dataset = _make_task_dataset(n_samples=10)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        db = TaskFingerprintDatabase()
        db.register_task(
            0, model, dataset, device, num_samples=10,
        )
        query = Counter({12345: 5, 67890: 3})
        best, scores = db.match(query, 1)
        assert best == 0

    # -----------------------------------------------------------------
    def test_empty_query(self):
        """Empty query returns task 0 with zero scores."""
        model = _make_dummy_model()
        device = torch.device('cpu')
        dataset = _make_task_dataset(n_samples=10)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        db = TaskFingerprintDatabase()
        db.register_task(
            0, model, dataset, device, num_samples=10,
        )
        best, scores = db.match(Counter(), 1)
        assert best == 0
        assert all(s == 0.0 for s in scores)
