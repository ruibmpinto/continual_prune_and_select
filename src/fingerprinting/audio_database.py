"""In-memory fingerprint database and matching engine.

Implements the back-end of the Shazam pipeline: storing
fingerprint hashes per song and matching query fingerprints
via the time-offset histogram method.

Classes
-------
FingerprintDatabase
    In-memory fingerprint store with insert and query.

Notes
-----
Based on Wang (2003) and Lennevi (2024).
"""

#
#                                                                Modules
# =====================================================================
# Standard
from collections import defaultdict

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
class FingerprintDatabase:
    """In-memory fingerprint database with offset histogram
    matching.

    Attributes
    ----------
    _db : dict
        Mapping hash_value -> list of (song_id, timestamp).
    _songs : set
        Set of registered song identifiers.

    Methods
    -------
    insert(song_id, hashes)
        Add fingerprint hashes for a song.
    query(hashes, min_certainty=1.5)
        Match query hashes against the database.
    """

    def __init__(self):
        """Constructor."""
        self._db = defaultdict(list)
        self._songs = set()

    # -----------------------------------------------------------------
    def insert(self, song_id, hashes):
        """Insert fingerprint hashes for a song.

        Parameters
        ----------
        song_id : {str, int}
            Identifier for the song.
        hashes : list[PeakPair]
            Output from compute_hashes.
        """
        self._songs.add(song_id)
        for pp in hashes:
            self._db[pp.hash_value].append(
                (song_id, pp.anchor_time),
            )

    # -----------------------------------------------------------------
    @property
    def num_songs(self):
        """Return number of registered songs.

        Returns
        -------
        n : int
            Number of distinct song identifiers.
        """
        return len(self._songs)

    # -----------------------------------------------------------------
    def query(self, hashes, min_certainty=1.5,
              min_score=3):
        """Match query hashes against the database.

        For each query hash, looks up all matching entries.
        Groups matches by song_id, computes time-offset
        histograms, and returns the song with the tallest
        histogram peak.

        Parameters
        ----------
        hashes : list[PeakPair]
            Fingerprint of the query audio.
        min_certainty : float, default=1.5
            Minimum ratio of best to second-best score
            to accept a match.
        min_score : int, default=3
            Minimum histogram peak height to accept
            a match. Prevents false positives from
            single hash collisions.

        Returns
        -------
        best_match : {str, int, None}
            Song ID of best match, or None if no match
            exceeds thresholds.
        score : int
            Peak height of the offset histogram for
            the best match.
        certainty : float
            Ratio best_score / second_best_score.
            Returns float('inf') if only one candidate.
        """
        # Collect (song_id, offset) for all matching hashes
        song_offsets = defaultdict(list)
        for pp in hashes:
            entries = self._db.get(pp.hash_value, [])
            for song_id, db_time in entries:
                offset = db_time - pp.anchor_time
                song_offsets[song_id].append(offset)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # No matches at all
        if not song_offsets:
            return None, 0, 0.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For each song, find the peak of its offset histogram
        scores = {}
        for song_id, offsets in song_offsets.items():
            scores[song_id] = _histogram_peak(offsets)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Sort by score descending
        ranked = sorted(
            scores.items(), key=lambda x: x[1], reverse=True,
        )
        best_id, best_score = ranked[0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute certainty
        if len(ranked) == 1:
            certainty = float('inf')
        else:
            second_score = ranked[1][1]
            if second_score == 0:
                certainty = float('inf')
            else:
                certainty = best_score / second_score
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if best_score < min_score:
            return None, best_score, certainty
        if certainty < min_certainty:
            return None, best_score, certainty
        return best_id, best_score, certainty
# =====================================================================


# =====================================================================
def _histogram_peak(offsets):
    """Find the peak count in an offset histogram.

    Parameters
    ----------
    offsets : list[int]
        Time offset values to histogram.

    Returns
    -------
    peak_count : int
        Count of the most common offset value.
    """
    if not offsets:
        return 0
    # Use sorting-based counting (faster than Counter for
    # large sparse histograms)
    offsets_sorted = sorted(offsets)
    max_count = 1
    current_count = 1
    for i in range(1, len(offsets_sorted)):
        if offsets_sorted[i] == offsets_sorted[i - 1]:
            current_count += 1
            if current_count > max_count:
                max_count = current_count
        else:
            current_count = 1
    return max_count
