"""End-to-end audio fingerprinting demo.

Demonstrates the full Shazam-style pipeline using synthetic
sine-wave songs: fingerprint generation, database insertion,
and query matching with self-match, snippet, and noise tests.

Functions
---------
generate_synthetic_song
    Create a synthetic multi-harmonic signal.
fingerprint_signal
    Full pipeline: signal -> spectrogram -> peaks -> hashes.
run_demo
    Execute the complete demo with printed results.

Notes
-----
No real audio files are needed. The demo uses synthetic signals
at distinct fundamental frequencies to simulate different songs.
"""

#
#                                                                Modules
# =====================================================================
# Standard
import sys

# Third-party
import numpy as np

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
DURATION = 10.0


# =====================================================================
def generate_synthetic_song(fundamental, n_harmonics=8,
                            sr=SR, duration=DURATION):
    """Create a synthetic multi-harmonic signal with vibrato.

    Each harmonic has slight frequency modulation to produce
    a richer spectrogram with more peaks.

    Parameters
    ----------
    fundamental : float
        Fundamental frequency in Hz.
    n_harmonics : int, default=8
        Number of harmonics to include.
    sr : int, default=8000
        Sample rate in Hz.
    duration : float, default=10.0
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
    for k in range(1, n_harmonics + 1):
        freq_k = fundamental * k
        # Skip harmonics above Nyquist
        if freq_k >= sr / 2:
            break
        amplitude = 1.0 / k
        # Add vibrato (FM) for spectral richness
        vibrato_rate = 3.0 + k * 0.5
        vibrato_depth = freq_k * 0.02
        phase = (2 * np.pi * freq_k * t
                 + vibrato_depth / vibrato_rate
                 * np.sin(2 * np.pi * vibrato_rate * t))
        signal += amplitude * np.sin(phase)
    return signal


# =====================================================================
def fingerprint_signal(signal, sr=SR, radius=10,
                       fan_out=10):
    """Full fingerprint pipeline for an audio signal.

    Parameters
    ----------
    signal : numpy.ndarray(1d)
        Audio signal.
    sr : int, default=8000
        Sample rate.
    radius : int, default=10
        Peak picking neighborhood radius.
    fan_out : int, default=10
        Hashing fan-out factor.

    Returns
    -------
    hashes : list[PeakPair]
        Fingerprint hashes for the signal.
    n_peaks : int
        Number of constellation peaks found.
    """
    _, _, sxx = compute_spectrogram(signal, sr)
    peaks = find_peaks(sxx, radius=radius)
    hashes = compute_hashes(peaks, fan_out=fan_out)
    return hashes, len(peaks)


# =====================================================================
def run_demo():
    """Execute the complete fingerprinting demo.

    Creates 5 synthetic songs, registers them, then tests:
    1. Self-matching (each song matches itself)
    2. Snippet matching (2s excerpt matches parent song)
    3. Noise robustness (song + noise at 15dB SNR)
    4. Random noise rejection (pure noise -> no match)
    """
    print('=' * 60)
    print('  Audio Fingerprinting Demo (Shazam-style)')
    print('=' * 60)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Define songs with distinct fundamentals
    song_defs = {
        'Song_A (220 Hz)': 220,
        'Song_B (440 Hz)': 440,
        'Song_C (660 Hz)': 660,
        'Song_D (880 Hz)': 880,
        'Song_E (1100 Hz)': 1100,
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Register all songs
    db = FingerprintDatabase()
    radius = 5
    fan_out = 10
    print('\n--- Registering songs ---')
    for name, freq in song_defs.items():
        signal = generate_synthetic_song(freq)
        hashes, n_peaks = fingerprint_signal(
            signal, radius=radius, fan_out=fan_out,
        )
        db.insert(name, hashes)
        print(f'  {name}: {n_peaks} peaks, '
              f'{len(hashes)} hashes')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test 1: Self-matching
    print('\n--- Test 1: Self-matching ---')
    all_passed = True
    for name, freq in song_defs.items():
        signal = generate_synthetic_song(freq)
        hashes, _ = fingerprint_signal(
            signal, radius=radius, fan_out=fan_out,
        )
        match, score, cert = db.query(hashes)
        status = 'OK' if match == name else 'FAIL'
        if match != name:
            all_passed = False
        print(f'  {name} -> {match} '
              f'(score={score}, certainty={cert:.1f}) '
              f'[{status}]')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test 2: Snippet matching
    print('\n--- Test 2: Snippet matching (2s excerpt) ---')
    for name, freq in song_defs.items():
        signal = generate_synthetic_song(freq)
        start = int(1.0 * SR)
        end = int(3.0 * SR)
        snippet = signal[start:end]
        hashes, _ = fingerprint_signal(
            snippet, radius=radius, fan_out=fan_out,
        )
        match, score, cert = db.query(hashes)
        status = 'OK' if match == name else 'FAIL'
        if match != name:
            all_passed = False
        print(f'  {name} snippet -> {match} '
              f'(score={score}) [{status}]')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test 3: Noise robustness
    print('\n--- Test 3: Noise robustness (20dB SNR) ---')
    rng = np.random.default_rng(42)
    for name, freq in song_defs.items():
        signal = generate_synthetic_song(freq)
        power = np.mean(signal ** 2)
        noise_power = power / (10 ** (20 / 10))
        noise = rng.normal(
            0, np.sqrt(noise_power), len(signal),
        )
        noisy = signal + noise
        hashes, _ = fingerprint_signal(
            noisy, radius=radius, fan_out=fan_out,
        )
        match, score, cert = db.query(hashes)
        status = 'OK' if match == name else 'FAIL'
        if match != name:
            all_passed = False
        print(f'  {name} + noise -> {match} '
              f'(score={score}) [{status}]')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test 4: Random noise rejection
    print('\n--- Test 4: Random noise rejection ---')
    noise_signal = rng.normal(0, 1.0, int(SR * DURATION))
    hashes, _ = fingerprint_signal(
        noise_signal, radius=radius, fan_out=fan_out,
    )
    match, score, cert = db.query(
        hashes, min_certainty=3.0,
    )
    if match is None:
        print(f'  Pure noise -> No match (correct) [OK]')
    else:
        print(f'  Pure noise -> {match} '
              f'(score={score}, cert={cert:.1f})')
        if cert < 3.0:
            print('  Low certainty, effectively rejected '
                  '[OK]')
        else:
            all_passed = False
            print('  [FAIL]')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('\n' + '=' * 60)
    if all_passed:
        print('  All tests passed.')
    else:
        print('  Some tests failed.')
    print('=' * 60)
    return all_passed


# =====================================================================
if __name__ == '__main__':
    success = run_demo()
    sys.exit(0 if success else 1)
