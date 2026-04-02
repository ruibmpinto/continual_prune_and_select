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
        self._db = TaskFingerprintDatabase(
            layer_names=self._layer_names,
            radius=radius,
            top_k=top_k,
            fan_out=fan_out,
        )

    # -----------------------------------------------------------------
    def register_task(self, task_id, dataset,
                      num_samples=None):
        """Build fingerprint DB for one task.

        Sets the model to the specified task, then
        fingerprints training data.

        Parameters
        ----------
        task_id : int
            Task identifier.
        dataset : torch.utils.data.Dataset
            Training dataset for this task.
        num_samples : int, default=None
            Number of samples to fingerprint. Uses
            constructor default if None.
        """
        if num_samples is None:
            num_samples = self._num_register_samples
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self._set_task_fn(self._model, task_id)
        self._model.eval()
        self._db.register_task(
            task_id, self._model, dataset,
            self._device, num_samples=num_samples,
        )

    # -----------------------------------------------------------------
    def select_task(self, x, num_learned):
        """Select best task for input batch.

        Runs forward pass through each task's subnetwork,
        computes fingerprint, and matches against stored
        task fingerprints.

        Parameters
        ----------
        x : torch.Tensor
            Input batch (B, C, H, W).
        num_learned : int
            Number of tasks learned so far.

        Returns
        -------
        task_id : int
            Selected task ID.
        """
        self._model.eval()
        best_task = 0
        best_score = -1.0
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for task_id in range(num_learned):
            self._set_task_fn(self._model, task_id)
            query_hashes = fingerprint_batch(
                self._model, x, self._device,
                layer_names=self._layer_names,
            )
            _, scores = self._db.match(
                query_hashes, num_learned,
            )
            if scores[task_id] > best_score:
                best_score = scores[task_id]
                best_task = task_id
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

        Parameters
        ----------
        filepath : str
            Input file path.
        """
        self._db.load(filepath)
