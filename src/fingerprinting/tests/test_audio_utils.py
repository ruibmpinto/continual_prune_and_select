"""Tests for audio_utils module.

Tests
-----
test_spectrogram_shape
    Verify output dimensions for a known signal.
test_spectrogram_frequency_content
    Verify dominant frequency appears at expected bin.
test_find_peaks_synthetic
    Verify peaks are found at known locations.
test_find_peaks_sorted_by_time
    Verify peaks are returned in time order.
test_find_peaks_empty_spectrogram
    Verify no peaks returned for silence.
"""

#
#                                                                Modules
# =====================================================================
# Standard
import math

# Third-party
import numpy as np
import pytest

# Local
from fingerprinting.audio_utils import (
    compute_spectrogram, find_peaks, Peak,
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

SR = 8000
DURATION = 2.0


def _make_sine(freq, sr=SR, duration=DURATION):
    """Generate a pure sine wave."""
    t = np.linspace(
        0, duration, int(sr * duration), endpoint=False,
    )
    return np.sin(2 * np.pi * freq * t)


# =====================================================================
class TestComputeSpectrogram:
    """Tests for compute_spectrogram."""

    # -----------------------------------------------------------------
    def test_spectrogram_shape(self):
        """Output has correct frequency and time dimensions."""
        signal = _make_sine(440)
        freqs, times, sxx = compute_spectrogram(signal, SR)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Frequency bins = window_size/2 + 1
        assert len(freqs) == 129
        # Time bins depend on signal length and overlap
        assert len(times) > 0
        assert sxx.shape == (len(freqs), len(times))

    # -----------------------------------------------------------------
    def test_spectrogram_frequency_content(self):
        """Dominant frequency bin matches the input sine."""
        signal = _make_sine(440)
        freqs, times, sxx = compute_spectrogram(signal, SR)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Average power across time, find peak frequency
        avg_power = sxx.mean(axis=1)
        peak_bin = np.argmax(avg_power)
        peak_freq = freqs[peak_bin]
        # Allow tolerance of one frequency bin width
        freq_resolution = freqs[1] - freqs[0]
        assert abs(peak_freq - 440) < 2 * freq_resolution


# =====================================================================
class TestFindPeaks:
    """Tests for find_peaks."""

    # -----------------------------------------------------------------
    def test_find_peaks_synthetic(self):
        """Peaks found at known bright spots in a synthetic
        spectrogram.
        """
        sxx = np.zeros((64, 100))
        # Place bright spots
        sxx[10, 20] = 10.0
        sxx[30, 50] = 8.0
        sxx[50, 80] = 12.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        peaks = find_peaks(sxx, radius=5)
        peak_positions = set(
            (p.time, p.freq) for p in peaks
        )
        assert (20, 10) in peak_positions
        assert (50, 30) in peak_positions
        assert (80, 50) in peak_positions

    # -----------------------------------------------------------------
    def test_find_peaks_sorted_by_time(self):
        """Peaks are sorted by time index."""
        signal = _make_sine(440) + 0.5 * _make_sine(880)
        _, _, sxx = compute_spectrogram(signal, SR)
        peaks = find_peaks(sxx, radius=10)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        times = [p.time for p in peaks]
        assert times == sorted(times)

    # -----------------------------------------------------------------
    def test_find_peaks_empty_spectrogram(self):
        """No peaks in a zero spectrogram."""
        sxx = np.zeros((64, 100))
        peaks = find_peaks(sxx, radius=10)
        assert len(peaks) == 0

    # -----------------------------------------------------------------
    def test_find_peaks_density(self):
        """Peak count scales inversely with radius squared."""
        signal = _make_sine(440) + 0.3 * _make_sine(1200)
        _, _, sxx = compute_spectrogram(signal, SR)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        peaks_small = find_peaks(sxx, radius=5)
        peaks_large = find_peaks(sxx, radius=15)
        # Larger radius should produce fewer peaks
        assert len(peaks_large) < len(peaks_small)

    # -----------------------------------------------------------------
    def test_peak_namedtuple(self):
        """Peak has time and freq attributes."""
        p = Peak(time=5, freq=10)
        assert p.time == 5
        assert p.freq == 10
