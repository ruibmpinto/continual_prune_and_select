"""Tests for audio_demo module.

Tests
-----
test_generate_synthetic_song
    Signal has correct length and non-zero content.
test_fingerprint_signal
    Pipeline produces non-empty hashes.
test_run_demo
    Full demo executes and passes all checks.
"""

#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
import pytest

# Local
from fingerprinting.audio_demo import (
    generate_synthetic_song, fingerprint_signal, run_demo,
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


# =====================================================================
class TestGenerateSyntheticSong:
    """Tests for generate_synthetic_song."""

    # -----------------------------------------------------------------
    def test_length(self):
        """Signal has expected number of samples."""
        signal = generate_synthetic_song(440, sr=8000,
                                         duration=2.0)
        assert len(signal) == 16000

    # -----------------------------------------------------------------
    def test_nonzero(self):
        """Signal is not all zeros."""
        signal = generate_synthetic_song(440)
        assert np.any(signal != 0)

    # -----------------------------------------------------------------
    def test_harmonics(self):
        """More harmonics produce richer signal."""
        s1 = generate_synthetic_song(440, n_harmonics=1)
        s5 = generate_synthetic_song(440, n_harmonics=5)
        # RMS of 5-harmonic signal should be larger
        rms1 = np.sqrt(np.mean(s1 ** 2))
        rms5 = np.sqrt(np.mean(s5 ** 2))
        assert rms5 > rms1


# =====================================================================
class TestFingerprintSignal:
    """Tests for fingerprint_signal."""

    # -----------------------------------------------------------------
    def test_produces_hashes(self):
        """Pipeline outputs non-empty hash list."""
        signal = generate_synthetic_song(440)
        hashes, n_peaks = fingerprint_signal(signal)
        assert len(hashes) > 0
        assert n_peaks > 0

    # -----------------------------------------------------------------
    def test_different_songs_different_hashes(self):
        """Different fundamentals yield different hash sets."""
        s1 = generate_synthetic_song(440)
        s2 = generate_synthetic_song(880)
        h1, _ = fingerprint_signal(s1)
        h2, _ = fingerprint_signal(s2)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        set1 = set(pp.hash_value for pp in h1)
        set2 = set(pp.hash_value for pp in h2)
        # Some overlap is possible, but not complete
        overlap = len(set1 & set2)
        total = len(set1 | set2)
        jaccard = overlap / total if total > 0 else 0
        assert jaccard < 0.5


# =====================================================================
class TestRunDemo:
    """Tests for run_demo."""

    # -----------------------------------------------------------------
    def test_demo_passes(self):
        """Full demo runs without errors and passes."""
        result = run_demo()
        assert result is True
