"""Tests for audio_database module.

Tests
-----
test_self_matching
    A song matches itself with high certainty.
test_snippet_matching
    A snippet from a song matches the full song.
test_noise_robustness
    Song with added noise still matches.
test_no_false_positive
    Random noise does not match any registered song.
test_multiple_songs_discrimination
    Correct song is identified among multiple.
test_empty_database
    Query against empty database returns None.
"""

#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
import pytest

# Local
from fingerprinting.audio_utils import (
    compute_spectrogram, find_peaks,
)
from fingerprinting.audio_hasher import compute_hashes
from fingerprinting.audio_database import FingerprintDatabase

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
DURATION = 5.0


def _make_song(freqs, sr=SR, duration=DURATION):
    """Generate a synthetic song as sum of sines.

    Parameters
    ----------
    freqs : list[float]
        Frequencies of constituent sine waves.
    sr : int
        Sample rate.
    duration : float
        Duration in seconds.

    Returns
    -------
    signal : numpy.ndarray(1d)
        Synthesized audio signal.
    """
    t = np.linspace(
        0, duration, int(sr * duration), endpoint=False,
    )
    signal = np.zeros_like(t)
    for i, f in enumerate(freqs):
        signal += (1.0 / (i + 1)) * np.sin(
            2 * np.pi * f * t
        )
    return signal


def _fingerprint(signal, sr=SR):
    """Full fingerprint pipeline for a signal.

    Returns
    -------
    hashes : list[PeakPair]
        Fingerprint hashes.
    """
    _, _, sxx = compute_spectrogram(signal, sr)
    peaks = find_peaks(sxx, radius=10)
    return compute_hashes(peaks, fan_out=10)


def _build_db_with_songs():
    """Build a database with 3 distinct synthetic songs.

    Returns
    -------
    db : FingerprintDatabase
        Database with songs registered.
    songs : dict
        Mapping song_id -> signal array.
    """
    db = FingerprintDatabase()
    songs = {
        'song_A': _make_song([440, 880, 1320]),
        'song_B': _make_song([550, 1100, 1650]),
        'song_C': _make_song([330, 660, 990]),
    }
    for song_id, signal in songs.items():
        hashes = _fingerprint(signal)
        db.insert(song_id, hashes)
    return db, songs


# =====================================================================
class TestFingerprintDatabase:
    """Tests for FingerprintDatabase."""

    # -----------------------------------------------------------------
    def test_self_matching(self):
        """A song matches itself with high certainty."""
        db, songs = _build_db_with_songs()
        query_hashes = _fingerprint(songs['song_A'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        match, score, certainty = db.query(query_hashes)
        assert match == 'song_A'
        assert score > 0
        assert certainty > 5.0

    # -----------------------------------------------------------------
    def test_snippet_matching(self):
        """A time snippet from a song matches the full song."""
        db, songs = _build_db_with_songs()
        # Take middle 2 seconds of song_B
        start = int(1.5 * SR)
        end = int(3.5 * SR)
        snippet = songs['song_B'][start:end]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        query_hashes = _fingerprint(snippet)
        match, score, certainty = db.query(query_hashes)
        assert match == 'song_B'

    # -----------------------------------------------------------------
    def test_noise_robustness(self):
        """Song with moderate noise (20dB SNR) still matches."""
        db, songs = _build_db_with_songs()
        signal = songs['song_C']
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Add noise at 20dB SNR
        rng = np.random.default_rng(42)
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (20 / 10))
        noise = rng.normal(
            0, np.sqrt(noise_power), len(signal),
        )
        noisy_signal = signal + noise
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        query_hashes = _fingerprint(noisy_signal)
        match, score, certainty = db.query(query_hashes)
        assert match == 'song_C'

    # -----------------------------------------------------------------
    def test_no_false_positive(self):
        """Pure random noise does not match any song."""
        db, _ = _build_db_with_songs()
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 1.0, int(SR * DURATION))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        query_hashes = _fingerprint(noise)
        match, score, certainty = db.query(
            query_hashes, min_certainty=3.0,
        )
        # Either no match or very low certainty
        if match is not None:
            assert certainty < 3.0 or score < 5

    # -----------------------------------------------------------------
    def test_multiple_songs_discrimination(self):
        """Each song matches itself, not others."""
        db, songs = _build_db_with_songs()
        for song_id, signal in songs.items():
            query_hashes = _fingerprint(signal)
            match, _, _ = db.query(query_hashes)
            assert match == song_id

    # -----------------------------------------------------------------
    def test_empty_database(self):
        """Query against empty database returns None."""
        db = FingerprintDatabase()
        signal = _make_song([440])
        query_hashes = _fingerprint(signal)
        match, score, certainty = db.query(query_hashes)
        assert match is None
        assert score == 0

    # -----------------------------------------------------------------
    def test_num_songs(self):
        """num_songs property tracks registered songs."""
        db, _ = _build_db_with_songs()
        assert db.num_songs == 3
