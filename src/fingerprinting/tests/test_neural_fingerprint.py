"""Tests for neural_fingerprint module.

Tests
-----
test_find_activation_peaks_synthetic
    Known peaks found in hand-crafted 3D tensor.
test_find_activation_peaks_top_k
    Top-k limiting works correctly.
test_hash_consistency
    Same activation produces same hashes.
test_hash_sensitivity
    Different activations produce different hashes.
test_extract_activation_maps
    Hook extraction produces correct shapes.
test_fingerprint_batch
    Full pipeline produces non-empty Counter.
"""

#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
import pytest
import torch
import torch.nn as nn

# Local
from fingerprinting.neural_fingerprint import (
    find_activation_peaks,
    hash_activation_peaks,
    extract_activation_maps,
    fingerprint_batch,
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
    """Create a small CNN with named layers for testing.

    Returns
    -------
    model : nn.Module
        A simple model with layer2 and layer3 attributes.
    """
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


# =====================================================================
class TestFindActivationPeaks:
    """Tests for find_activation_peaks."""

    # -----------------------------------------------------------------
    def test_synthetic_peaks(self):
        """Known peaks found in hand-crafted tensor."""
        act = np.zeros((4, 16, 16))
        # Place bright spots
        act[0, 4, 4] = 10.0
        act[1, 8, 8] = 8.0
        act[2, 12, 12] = 12.0
        act[3, 2, 14] = 6.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        peaks = find_activation_peaks(act, radius=3, top_k=10)
        peak_positions = set(
            (p[0], p[1], p[2]) for p in peaks
        )
        assert (0, 4, 4) in peak_positions
        assert (1, 8, 8) in peak_positions
        assert (2, 12, 12) in peak_positions
        assert (3, 2, 14) in peak_positions

    # -----------------------------------------------------------------
    def test_top_k_limit(self):
        """Top-k limits the number of returned peaks."""
        act = np.random.default_rng(42).random((8, 16, 16))
        peaks = find_activation_peaks(
            act, radius=2, top_k=5,
        )
        assert len(peaks) <= 5

    # -----------------------------------------------------------------
    def test_sorted_by_magnitude(self):
        """Peaks are sorted by magnitude descending."""
        act = np.random.default_rng(42).random((4, 16, 16))
        peaks = find_activation_peaks(
            act, radius=2, top_k=20,
        )
        mags = [p[3] for p in peaks]
        assert mags == sorted(mags, reverse=True)

    # -----------------------------------------------------------------
    def test_zero_activation(self):
        """No peaks in a zero activation."""
        act = np.zeros((4, 8, 8))
        peaks = find_activation_peaks(act, radius=2)
        assert len(peaks) == 0


# =====================================================================
class TestHashActivationPeaks:
    """Tests for hash_activation_peaks."""

    # -----------------------------------------------------------------
    def test_consistency(self):
        """Same peaks produce same hashes."""
        peaks = [
            (0, 4, 4, 10.0),
            (1, 8, 8, 8.0),
            (2, 12, 12, 12.0),
            (3, 2, 14, 6.0),
        ]
        h1 = hash_activation_peaks(peaks, fan_out=3)
        h2 = hash_activation_peaks(peaks, fan_out=3)
        assert h1 == h2

    # -----------------------------------------------------------------
    def test_sensitivity(self):
        """Different peaks produce different hashes."""
        peaks_a = [
            (0, 4, 4, 10.0),
            (1, 8, 8, 8.0),
            (2, 12, 12, 12.0),
        ]
        peaks_b = [
            (0, 2, 2, 5.0),
            (3, 10, 6, 7.0),
            (1, 14, 14, 9.0),
        ]
        h_a = hash_activation_peaks(peaks_a, fan_out=2)
        h_b = hash_activation_peaks(peaks_b, fan_out=2)
        # At least some hashes should differ
        overlap = sum(
            (h_a & h_b).values()
        )
        total = sum(
            (h_a | h_b).values()
        )
        assert overlap < total

    # -----------------------------------------------------------------
    def test_single_peak(self):
        """Single peak produces empty hashes."""
        peaks = [(0, 4, 4, 10.0)]
        h = hash_activation_peaks(peaks, fan_out=3)
        assert len(h) == 0

    # -----------------------------------------------------------------
    def test_empty_peaks(self):
        """Empty peaks produce empty hashes."""
        h = hash_activation_peaks([], fan_out=3)
        assert len(h) == 0


# =====================================================================
class TestExtractActivationMaps:
    """Tests for extract_activation_maps."""

    # -----------------------------------------------------------------
    def test_correct_shapes(self):
        """Activations have expected shapes."""
        model = _make_dummy_model()
        device = torch.device('cpu')
        x = torch.randn(4, 3, 32, 32)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        acts = extract_activation_maps(
            model, x, device,
            layer_names=['layer2', 'layer3'],
        )
        assert 'layer2' in acts
        assert 'layer3' in acts
        # layer2: Conv2d(8->16, stride=2) on 32x32 -> 16x16
        assert acts['layer2'].shape == (4, 16, 16, 16)
        # layer3: Conv2d(16->32, stride=2) on 16x16 -> 8x8
        assert acts['layer3'].shape == (4, 32, 8, 8)

    # -----------------------------------------------------------------
    def test_hooks_removed(self):
        """No hooks remain after extraction."""
        model = _make_dummy_model()
        device = torch.device('cpu')
        x = torch.randn(2, 3, 32, 32)
        _ = extract_activation_maps(
            model, x, device,
            layer_names=['layer2'],
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Check no hooks remain on layer2
        hooks = model.layer2._forward_hooks
        assert len(hooks) == 0


# =====================================================================
class TestFingerprintBatch:
    """Tests for fingerprint_batch."""

    # -----------------------------------------------------------------
    def test_produces_hashes(self):
        """Pipeline produces non-empty Counter."""
        model = _make_dummy_model()
        device = torch.device('cpu')
        x = torch.randn(4, 3, 32, 32)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        hashes = fingerprint_batch(
            model, x, device,
            layer_names=['layer2', 'layer3'],
        )
        assert isinstance(hashes, dict)
        assert len(hashes) > 0

    # -----------------------------------------------------------------
    def test_deterministic(self):
        """Same input produces same hashes."""
        model = _make_dummy_model()
        model.eval()
        device = torch.device('cpu')
        x = torch.randn(2, 3, 32, 32)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        h1 = fingerprint_batch(
            model, x, device,
            layer_names=['layer2'],
        )
        h2 = fingerprint_batch(
            model, x, device,
            layer_names=['layer2'],
        )
        assert h1 == h2
