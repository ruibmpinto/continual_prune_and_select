"""Task fingerprint database and matching engine.

Stores per-task fingerprint hash distributions and matches
query fingerprints against stored tasks using hash-set
overlap scoring.

Classes
-------
TaskFingerprintDatabase
    Per-task fingerprint store with register and match.

Notes
-----
Adapted from the Shazam offset-histogram method. Since
neural activations lack a temporal axis, matching uses
hash occurrence overlap instead of time-offset alignment.
"""

#
#                                                                Modules
# =====================================================================
# Standard
from collections import Counter

# Third-party
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local
from fingerprinting.neural_fingerprint import (
    fingerprint_batch,
    DEFAULT_LAYERS,
    DEFAULT_RADIUS,
    DEFAULT_TOP_K,
    DEFAULT_FAN_OUT,
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
class TaskFingerprintDatabase:
    """Per-task fingerprint store with overlap matching.

    Attributes
    ----------
    _task_hashes : dict
        Mapping task_id -> Counter of hash occurrences.
    _num_samples : dict
        Mapping task_id -> number of samples fingerprinted.

    Methods
    -------
    register_task(task_id, model, dataset, device, ...)
        Build fingerprint DB for a task from training data.
    match(query_hashes, num_learned)
        Match query fingerprint against stored tasks.
    """

    def __init__(self, layer_names=None,
                 radius=DEFAULT_RADIUS,
                 top_k=DEFAULT_TOP_K,
                 fan_out=DEFAULT_FAN_OUT):
        """Constructor.

        Parameters
        ----------
        layer_names : list[str], default=None
            Layers to extract activations from.
        radius : int, default=3
            Peak picking radius.
        top_k : int, default=50
            Max peaks per sample per layer.
        fan_out : int, default=5
            Hashing fan-out.
        """
        self._task_hashes = {}
        self._num_samples = {}
        self._layer_names = (
            layer_names if layer_names is not None
            else DEFAULT_LAYERS
        )
        self._radius = radius
        self._top_k = top_k
        self._fan_out = fan_out

    # -----------------------------------------------------------------
    @property
    def num_tasks(self):
        """Return number of registered tasks.

        Returns
        -------
        n : int
            Number of registered tasks.
        """
        return len(self._task_hashes)

    # -----------------------------------------------------------------
    def register_task(self, task_id, model, dataset,
                      device, num_samples=500,
                      batch_size=64):
        """Build fingerprint DB for a task.

        Fingerprints up to num_samples training examples and
        accumulates hash occurrence counts.

        Parameters
        ----------
        task_id : int
            Task identifier.
        model : torch.nn.Module
            Model with task_id set and masks applied.
        dataset : torch.utils.data.Dataset
            Training dataset for this task.
        device : torch.device
            Computation device.
        num_samples : int, default=500
            Maximum training samples to fingerprint.
        batch_size : int, default=64
            Batch size for fingerprinting.
        """
        model.eval()
        task_counter = Counter()
        total_samples = 0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for batch_x, _ in loader:
            if total_samples >= num_samples:
                break
            remaining = num_samples - total_samples
            if batch_x.size(0) > remaining:
                batch_x = batch_x[:remaining]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            batch_hashes = fingerprint_batch(
                model, batch_x, device,
                layer_names=self._layer_names,
                radius=self._radius,
                top_k=self._top_k,
                fan_out=self._fan_out,
            )
            task_counter.update(batch_hashes)
            total_samples += batch_x.size(0)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._task_hashes[task_id] = task_counter
        self._num_samples[task_id] = total_samples

    # -----------------------------------------------------------------
    def match(self, query_hashes, num_learned):
        """Match query fingerprint against stored tasks.

        Computes hash-set overlap between the query and each
        stored task fingerprint. The task with the highest
        normalized overlap wins.

        Parameters
        ----------
        query_hashes : Counter
            Hash occurrence counts from the query batch.
        num_learned : int
            Number of tasks learned so far.

        Returns
        -------
        best_task : int
            Task ID with highest match score.
        scores : list[float]
            Per-task normalized overlap scores.
        """
        query_total = sum(query_hashes.values())
        if query_total == 0:
            return None, [0.0] * num_learned
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        scores = []
        for task_id in range(num_learned):
            if task_id not in self._task_hashes:
                scores.append(0.0)
                continue
            task_counter = self._task_hashes[task_id]
            # Overlap = sum of min counts for shared hashes
            overlap = 0
            for h, q_count in query_hashes.items():
                if h in task_counter:
                    overlap += min(
                        q_count, task_counter[h],
                    )
            # Normalize by query total
            scores.append(overlap / query_total)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        best_task = int(np.argmax(scores))
        return best_task, scores

    # -----------------------------------------------------------------
    def save(self, filepath):
        """Save database to file.

        Parameters
        ----------
        filepath : str
            Path to save the database.
        """
        data = {
            'task_hashes': dict(self._task_hashes),
            'num_samples': dict(self._num_samples),
            'layer_names': self._layer_names,
            'radius': self._radius,
            'top_k': self._top_k,
            'fan_out': self._fan_out,
        }
        torch.save(data, filepath)

    # -----------------------------------------------------------------
    def load(self, filepath):
        """Load database from file.

        Parameters
        ----------
        filepath : str
            Path to load the database from.
        """
        data = torch.load(filepath, weights_only=False)
        self._task_hashes = {
            k: Counter(v)
            for k, v in data['task_hashes'].items()
        }
        self._num_samples = data['num_samples']
        self._layer_names = data['layer_names']
        self._radius = data['radius']
        self._top_k = data['top_k']
        self._fan_out = data['fan_out']
