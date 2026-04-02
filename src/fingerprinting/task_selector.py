"""Integration adapter for fingerprint-based task selection.

Provides a drop-in replacement for the IS-based
select_subnetwork function used in the CPS ResNet pipeline.
Wraps neural fingerprinting and task fingerprint database.

Classes
-------
FingerprintTaskSelector
    Drop-in task selector using neural fingerprints.

Notes
-----
Designed for integration with resnet18_cifar100.py.
Requires the model to have layer2 and layer3 attributes
(nn.Sequential modules) and a set_task-compatible interface.
"""

#
#                                                                Modules
# =====================================================================
# Standard
import copy

# Third-party
import numpy as np
import torch

# Local
from fingerprinting.neural_fingerprint import (
    fingerprint_batch,
    DEFAULT_LAYERS,
    DEFAULT_RADIUS,
    DEFAULT_TOP_K,
    DEFAULT_FAN_OUT,
)
from fingerprinting.neural_database import (
    TaskFingerprintDatabase,
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
class FingerprintTaskSelector:
    """Drop-in task selector using neural fingerprints.

    Replaces select_subnetwork by matching input fingerprints
    against stored per-task fingerprint databases.

    Attributes
    ----------
    _model : torch.nn.Module
        The ResNet model.
    _device : torch.device
        Computation device.
    _db : TaskFingerprintDatabase
        Per-task fingerprint store.
    _set_task_fn : callable
        Function to switch task on the model.

    Methods
    -------
    register_task(task_id, dataset, num_samples)
        Build fingerprint DB for one task.
    select_task(x, num_learned)
        Select best task for input batch.
    save(filepath)
        Save fingerprint database to file.
    load(filepath)
        Load fingerprint database from file.
    """

    def __init__(self, model, device, set_task_fn,
                 layer_names=None, radius=DEFAULT_RADIUS,
                 top_k=DEFAULT_TOP_K,
                 fan_out=DEFAULT_FAN_OUT,
                 num_register_samples=500):
        """Constructor.

        Parameters
        ----------
        model : torch.nn.Module
            The trained ResNet model.
        device : torch.device
            Computation device.
        set_task_fn : callable
            Function(model, task_id) to switch active
            task/masks on the model.
        layer_names : list[str], default=None
            Layers to extract activations from.
            Defaults to ['layer2', 'layer3'].
        radius : int, default=3
            Peak picking radius.
        top_k : int, default=50
            Max peaks per sample per layer.
        fan_out : int, default=5
            Hashing fan-out.
        num_register_samples : int, default=500
            Default number of training samples for
            fingerprint registration.
        """
        self._model = model
        self._device = device
        self._set_task_fn = set_task_fn
        self._num_register_samples = num_register_samples
        self._layer_names = (
            layer_names if layer_names is not None
            else DEFAULT_LAYERS
        )
        self._radius = radius
        self._top_k = top_k
        self._fan_out = fan_out
        self._db = TaskFingerprintDatabase(
            layer_names=self._layer_names,
            radius=radius,
            top_k=top_k,
            fan_out=fan_out,
        )

    # -----------------------------------------------------------------
    def register_task(self, task_id, dataset,
                      num_samples=None,
                      deterministic_transform=None):
        """Build fingerprint DB for one task.

        Sets the model to the specified task, then
        fingerprints training data. If the dataset uses
        stochastic augmentations, a deterministic transform
        should be provided to ensure reproducible hashes.

        Parameters
        ----------
        task_id : int
            Task identifier.
        dataset : torch.utils.data.Dataset
            Training dataset for this task.
        num_samples : int, default=None
            Number of samples to fingerprint. Uses
            constructor default if None.
        deterministic_transform : callable, default=None
            If provided, temporarily replaces the dataset
            transform during registration to avoid
            stochastic augmentation noise. Restored after.
        """
        if num_samples is None:
            num_samples = self._num_register_samples
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Swap to deterministic transform if provided
        original_transform = None
        if (deterministic_transform is not None
                and hasattr(dataset, 'transform')):
            original_transform = dataset.transform
            dataset.transform = deterministic_transform
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._set_task_fn(self._model, task_id)
        self._model.eval()
        self._db.register_task(
            task_id, self._model, dataset,
            self._device, num_samples=num_samples,
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Restore original transform
        if original_transform is not None:
            dataset.transform = original_transform

    # -----------------------------------------------------------------
    def select_task(self, x, num_learned,
                    min_score=0.0, min_certainty=1.5):
        """Select best task for input batch.

        Runs forward pass through each task's subnetwork,
        computes fingerprint, and matches against stored
        task fingerprints. Abstains if best score is too
        low or the margin over the runner-up is too small.

        Parameters
        ----------
        x : torch.Tensor
            Input batch (B, C, H, W).
        num_learned : int
            Number of tasks learned so far.
        min_score : float, default=0.0
            Minimum Jaccard score to accept a match.
        min_certainty : float, default=1.5
            Minimum ratio of best to second-best score.
            Abstains on near-ties. Set to 1.0 to disable.

        Returns
        -------
        task_id : {int, None}
            Selected task ID, or None if the match does
            not meet score or certainty thresholds.
        """
        self._model.eval()
        task_scores = []
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for task_id in range(num_learned):
            self._set_task_fn(self._model, task_id)
            query_hashes = fingerprint_batch(
                self._model, x, self._device,
                layer_names=self._layer_names,
                radius=self._radius,
                top_k=self._top_k,
                fan_out=self._fan_out,
            )
            _, scores = self._db.match(
                query_hashes, num_learned,
            )
            task_scores.append(scores[task_id])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Find best and second-best
        ranked = sorted(
            range(len(task_scores)),
            key=lambda i: task_scores[i],
            reverse=True,
        )
        best_task = ranked[0]
        best_score = task_scores[best_task]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Absolute score threshold
        if best_score <= min_score:
            return None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Certainty ratio check (best / second-best)
        if len(ranked) > 1:
            second_score = task_scores[ranked[1]]
            if second_score > 0:
                certainty = best_score / second_score
                if certainty < min_certainty:
                    return None
        return best_task

    # -----------------------------------------------------------------
    def save(self, filepath):
        """Save fingerprint database to file.

        Parameters
        ----------
        filepath : str
            Output file path.
        """
        self._db.save(filepath)

    # -----------------------------------------------------------------
    def load(self, filepath):
        """Load fingerprint database from file.

        Restores hyperparameters from the saved database
        to ensure query-time fingerprinting matches the
        registration scheme.

        Parameters
        ----------
        filepath : str
            Input file path.
        """
        self._db.load(filepath)
        # Sync hyperparameters from loaded DB
        self._layer_names = self._db._layer_names
        self._radius = self._db._radius
        self._top_k = self._db._top_k
        self._fan_out = self._db._fan_out
