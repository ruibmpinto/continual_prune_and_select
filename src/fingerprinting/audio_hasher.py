"""Combinatorial hashing of constellation map peaks.

Implements the core Shazam fingerprinting step: pairing each
anchor peak with nearby forward peaks and computing a polynomial
rolling hash from (freq1, freq2, delta_time).

Classes
-------
PeakPair
    Stores hash value and anchor timestamp for a paired peak.

Functions
---------
compute_hashes
    Generate combinatorial hashes from constellation peaks.

Notes
-----
Based on Wang (2003) and Lennevi (2024).
"""

#
#                                                                Modules
# =====================================================================
# Standard
from collections import namedtuple

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================

PRIME1 = 52711
PRIME2 = 1000000007

PeakPair = namedtuple(
    'PeakPair',
    ['hash_value', 'anchor_time', 'freq1', 'freq2',
     'delta_time'],
)


# =====================================================================
def _hash_triplet(freq1, freq2, delta_time):
    """Compute polynomial rolling hash of a peak pair.

    Parameters
    ----------
    freq1 : int
        Frequency bin of the anchor peak.
    freq2 : int
        Frequency bin of the target peak.
    delta_time : int
        Time difference between target and anchor.

    Returns
    -------
    h : int
        Hash value in range [0, PRIME2).
    """
    h = (freq1
         + freq2 * PRIME1
         + delta_time * PRIME1 * PRIME1) % PRIME2
    return h


# =====================================================================
def compute_hashes(peaks, fan_out=10):
    """Generate combinatorial hashes from constellation peaks.

    For each anchor peak, pairs it with up to fan_out forward
    peaks sorted by proximity. Each pair produces one hash.

    Parameters
    ----------
    peaks : list[Peak]
        Sorted list of Peak(time, freq) from find_peaks.
    fan_out : int, default=10
        Maximum target peaks per anchor.

    Returns
    -------
    hashes : list[PeakPair]
        List of PeakPair namedtuples.
    """
    n = len(peaks)
    hashes = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(n):
        anchor = peaks[i]
        count = 0
        for j in range(i + 1, n):
            if count >= fan_out:
                break
            target = peaks[j]
            dt = target.time - anchor.time
            # Skip peaks at the same time
            if dt <= 0:
                continue
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            h = _hash_triplet(anchor.freq, target.freq, dt)
            hashes.append(PeakPair(
                hash_value=h,
                anchor_time=anchor.time,
                freq1=anchor.freq,
                freq2=target.freq,
                delta_time=dt,
            ))
            count += 1
    return hashes
