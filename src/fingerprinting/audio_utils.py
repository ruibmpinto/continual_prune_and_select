"""Spectrogram computation and constellation map extraction.

Implements the front-end of the Shazam fingerprinting pipeline:
STFT-based spectrogram computation and local-maximum peak picking
to produce a sparse constellation map.

Functions
---------
compute_spectrogram
    Compute STFT magnitude spectrogram from audio signal.
find_peaks
    Extract constellation map peaks via local max filtering.

Notes
-----
Based on Wang (2003) and Lennevi (2024).
"""

#
#                                                                Modules
# =====================================================================
# Standard
from collections import namedtuple

# Third-party
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import spectrogram

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================

Peak = namedtuple('Peak', ['time', 'freq'])


# =====================================================================
def compute_spectrogram(signal, sr, window_size=256, overlap=32):
    """Compute STFT magnitude spectrogram.

    Parameters
    ----------
    signal : numpy.ndarray(1d)
        Mono audio signal.
    sr : int
        Sample rate in Hz.
    window_size : int, default=256
        Hann window length in samples.
    overlap : int, default=32
        Overlap between consecutive frames in samples.

    Returns
    -------
    freqs : numpy.ndarray(1d)
        Frequency bin centers in Hz.
    times : numpy.ndarray(1d)
        Time bin centers in seconds.
    sxx : numpy.ndarray(2d)
        Magnitude spectrogram of shape (n_freqs, n_times).
    """
    freqs, times, sxx = spectrogram(
        signal, fs=sr, window='hann',
        nperseg=window_size, noverlap=overlap,
    )
    return freqs, times, sxx


# =====================================================================
def find_peaks(sxx, radius=10):
    """Extract constellation map peaks via local max filtering.

    A point is a peak if it equals the maximum value in its
    local neighborhood of size (2*radius+1) x (2*radius+1).

    Parameters
    ----------
    sxx : numpy.ndarray(2d)
        Magnitude spectrogram of shape (n_freqs, n_times).
    radius : int, default=10
        Neighborhood radius for the maximum filter.

    Returns
    -------
    peaks : list[Peak]
        List of Peak(time, freq) namedtuples sorted by time.
    """
    kernel_size = 2 * radius + 1
    local_max = maximum_filter(sxx, size=kernel_size)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # A peak is where the original equals the local max and is
    # nonzero (avoid silence regions)
    peak_mask = (sxx == local_max) & (sxx > 0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # np.argwhere returns (freq_idx, time_idx) pairs
    coords = np.argwhere(peak_mask)
    peaks = [Peak(time=int(c[1]), freq=int(c[0]))
             for c in coords]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort by time, then by frequency for deterministic order
    peaks.sort(key=lambda p: (p.time, p.freq))
    return peaks
