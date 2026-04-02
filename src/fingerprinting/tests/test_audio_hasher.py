"""Tests for audio_hasher module.

Tests
-----
test_hash_determinism
    Same peaks produce identical hashes.
test_hash_sensitivity
    Different peaks produce different hashes.
test_fan_out_limit
    Number of hashes respects fan_out parameter.
test_empty_peaks
    No peaks produce no hashes.
test_single_peak
    One peak produces no hashes (need pairs).
test_peak_pair_fields
    PeakPair has correct fields.
"""

#
#                                                                Modules
# =====================================================================
# Third-party
import pytest

# Local
from fingerprinting.audio_utils import Peak
from fingerprinting.audio_hasher import (
    compute_hashes, PeakPair, _hash_triplet,
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

SAMPLE_PEAKS = [
    Peak(time=0, freq=10),
    Peak(time=5, freq=20),
    Peak(time=10, freq=15),
    Peak(time=15, freq=30),
    Peak(time=20, freq=25),
]


# =====================================================================
class TestHashTriplet:
    """Tests for the internal hash function."""

    # -----------------------------------------------------------------
    def test_determinism(self):
        """Same inputs produce same hash."""
        h1 = _hash_triplet(10, 20, 5)
        h2 = _hash_triplet(10, 20, 5)
        assert h1 == h2

    # -----------------------------------------------------------------
    def test_sensitivity_freq1(self):
        """Different freq1 produces different hash."""
        h1 = _hash_triplet(10, 20, 5)
        h2 = _hash_triplet(11, 20, 5)
        assert h1 != h2

    # -----------------------------------------------------------------
    def test_sensitivity_freq2(self):
        """Different freq2 produces different hash."""
        h1 = _hash_triplet(10, 20, 5)
        h2 = _hash_triplet(10, 21, 5)
        assert h1 != h2

    # -----------------------------------------------------------------
    def test_sensitivity_dt(self):
        """Different delta_time produces different hash."""
        h1 = _hash_triplet(10, 20, 5)
        h2 = _hash_triplet(10, 20, 6)
        assert h1 != h2

    # -----------------------------------------------------------------
    def test_range(self):
        """Hash is non-negative and bounded."""
        h = _hash_triplet(100, 200, 50)
        assert 0 <= h < 1000000007


# =====================================================================
class TestComputeHashes:
    """Tests for compute_hashes."""

    # -----------------------------------------------------------------
    def test_hash_determinism(self):
        """Same peaks produce identical hash lists."""
        h1 = compute_hashes(SAMPLE_PEAKS, fan_out=3)
        h2 = compute_hashes(SAMPLE_PEAKS, fan_out=3)
        assert len(h1) == len(h2)
        for a, b in zip(h1, h2):
            assert a.hash_value == b.hash_value

    # -----------------------------------------------------------------
    def test_fan_out_limit(self):
        """Each anchor produces at most fan_out pairs."""
        hashes = compute_hashes(SAMPLE_PEAKS, fan_out=2)
        # First anchor (time=0) should pair with at most 2
        anchor_0_hashes = [
            h for h in hashes if h.anchor_time == 0
        ]
        assert len(anchor_0_hashes) <= 2

    # -----------------------------------------------------------------
    def test_total_hash_count(self):
        """Total hashes bounded by n_peaks * fan_out."""
        fan_out = 3
        hashes = compute_hashes(SAMPLE_PEAKS, fan_out=fan_out)
        assert len(hashes) <= len(SAMPLE_PEAKS) * fan_out

    # -----------------------------------------------------------------
    def test_empty_peaks(self):
        """No peaks produce no hashes."""
        hashes = compute_hashes([], fan_out=10)
        assert len(hashes) == 0

    # -----------------------------------------------------------------
    def test_single_peak(self):
        """One peak produces no hashes."""
        hashes = compute_hashes(
            [Peak(time=0, freq=10)], fan_out=10,
        )
        assert len(hashes) == 0

    # -----------------------------------------------------------------
    def test_peak_pair_fields(self):
        """PeakPair has expected fields."""
        hashes = compute_hashes(SAMPLE_PEAKS, fan_out=1)
        pp = hashes[0]
        assert hasattr(pp, 'hash_value')
        assert hasattr(pp, 'anchor_time')
        assert hasattr(pp, 'freq1')
        assert hasattr(pp, 'freq2')
        assert hasattr(pp, 'delta_time')
        assert pp.delta_time > 0

    # -----------------------------------------------------------------
    def test_forward_only(self):
        """All pairs have positive delta_time."""
        hashes = compute_hashes(SAMPLE_PEAKS, fan_out=10)
        for h in hashes:
            assert h.delta_time > 0
