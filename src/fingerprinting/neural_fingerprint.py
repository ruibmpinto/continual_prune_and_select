"""Activation-based peak extraction and combinatorial hashing.

Adapts the Shazam fingerprinting algorithm to neural network
activations. Feature maps from intermediate layers serve as
spectrograms; spatial local maxima serve as constellation
peaks; combinatorial hashing of peak pairs produces task
fingerprints.

Functions
---------
extract_activation_maps
    Capture intermediate activations via forward hooks.
find_activation_peaks
    Peak picking on 3D activation tensors.
hash_activation_peaks
    Combinatorial hashing of activation peak pairs.
fingerprint_batch
    Full pipeline: batch -> activations -> peaks -> hashes.

Notes
-----
Inspired by Wang (2003) Shazam algorithm, adapted for
neural activation maps in a continual learning context.
"""

#
#                                                                Modules
# =====================================================================
# Standard
from collections import Counter

# Third-party
import numpy as np
import torch
from scipy.ndimage import maximum_filter

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================

HASH_PRIME1 = 52711
HASH_PRIME2 = 1000000007
DEFAULT_LAYERS = ['layer2', 'layer3']
DEFAULT_RADIUS = 3
DEFAULT_TOP_K = 50
DEFAULT_FAN_OUT = 5
DEFAULT_N_MAG_BINS = 8


# =====================================================================
def extract_activation_maps(model, x, device,
                            layer_names=None):
    """Capture intermediate activations via forward hooks.

    Registers temporary forward hooks on the specified layers,
    runs a forward pass, then removes the hooks. Does not
    modify the model.

    Parameters
    ----------
    model : torch.nn.Module
        The network (with task_id already set).
    x : torch.Tensor
        Input batch of shape (B, C, H, W).
    device : torch.device
        Computation device.
    layer_names : list[str], default=None
        Layer attribute names to hook. Defaults to
        ['layer2', 'layer3'].

    Returns
    -------
    activations : dict
        Mapping layer_name -> numpy.ndarray of shape
        (B, C_out, H_out, W_out).
    """
    if layer_names is None:
        layer_names = DEFAULT_LAYERS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    captured = {}
    hooks = []

    def _make_hook(name):
        def hook_fn(module, input, output):
            captured[name] = output.detach().cpu().numpy()
        return hook_fn

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for name in layer_names:
        layer = getattr(model, name)
        h = layer.register_forward_hook(_make_hook(name))
        hooks.append(h)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    with torch.no_grad():
        x = x.to(device)
        _ = model(x)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for h in hooks:
        h.remove()
    return captured


# =====================================================================
def find_activation_peaks(activation, radius=DEFAULT_RADIUS,
                          top_k=DEFAULT_TOP_K):
    """Peak picking on a single-sample 3D activation tensor.

    For each channel, applies a spatial maximum filter and
    identifies local maxima. Returns the top_k peaks by
    magnitude.

    Parameters
    ----------
    activation : numpy.ndarray(3d)
        Activation of shape (C, H, W) for one sample.
    radius : int, default=3
        Spatial neighborhood radius for max filter.
    top_k : int, default=50
        Maximum number of peaks to retain.

    Returns
    -------
    peaks : list[tuple]
        List of (channel, h, w, magnitude) tuples, sorted
        by magnitude descending, at most top_k entries.
    """
    c, h_size, w_size = activation.shape
    kernel_size = 2 * radius + 1
    peaks = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for ch in range(c):
        channel_act = activation[ch]
        local_max = maximum_filter(
            channel_act, size=kernel_size,
        )
        peak_mask = (
            (channel_act == local_max)
            & (channel_act > 0)
        )
        coords = np.argwhere(peak_mask)
        for coord in coords:
            mag = float(channel_act[coord[0], coord[1]])
            peaks.append((ch, int(coord[0]),
                          int(coord[1]), mag))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sort by magnitude descending, keep top_k
    peaks.sort(key=lambda p: p[3], reverse=True)
    return peaks[:top_k]


# =====================================================================
def _quantize_magnitude(magnitude, bin_edges):
    """Quantize a magnitude value into a discrete bin.

    Parameters
    ----------
    magnitude : float
        Peak magnitude value.
    bin_edges : numpy.ndarray(1d)
        Bin edges from np.quantile.

    Returns
    -------
    bin_idx : int
        Quantized bin index.
    """
    idx = int(np.searchsorted(bin_edges, magnitude))
    return min(idx, len(bin_edges) - 1)


# =====================================================================
def hash_activation_peaks(peaks, fan_out=DEFAULT_FAN_OUT,
                          bin_edges=None,
                          n_mag_bins=DEFAULT_N_MAG_BINS):
    """Combinatorial hashing of activation peak pairs.

    Each anchor peak is paired with up to fan_out nearest
    peaks (by spatial distance). The hash encodes channel
    IDs, spatial offset, and magnitude bin difference.

    Parameters
    ----------
    peaks : list[tuple]
        List of (channel, h, w, magnitude) from
        find_activation_peaks.
    fan_out : int, default=5
        Maximum number of target peaks per anchor.
    bin_edges : numpy.ndarray(1d), default=None
        Magnitude quantization bin edges. If None,
        magnitudes are binned uniformly into n_mag_bins.
    n_mag_bins : int, default=8
        Number of magnitude bins if bin_edges is None.

    Returns
    -------
    hashes : Counter
        Counter mapping hash_value -> occurrence count.
    """
    if len(peaks) < 2:
        return Counter()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute bin edges from peak magnitudes if not provided
    if bin_edges is None:
        mags = np.array([p[3] for p in peaks])
        if mags.max() == mags.min():
            bin_edges = np.array([mags.min()])
        else:
            quantiles = np.linspace(
                0, 1, n_mag_bins + 1,
            )[1:-1]
            bin_edges = np.quantile(mags, quantiles)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Precompute spatial positions for distance
    positions = np.array(
        [[p[1], p[2]] for p in peaks], dtype=np.float32,
    )
    hashes = Counter()
    n = len(peaks)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(n):
        ch1, h1, w1, mag1 = peaks[i]
        mag_bin1 = _quantize_magnitude(mag1, bin_edges)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compute distances to all other peaks
        dists = np.sqrt(
            (positions[:, 0] - h1) ** 2
            + (positions[:, 1] - w1) ** 2,
        )
        dists[i] = float('inf')
        # Get fan_out nearest neighbors
        nearest_idx = np.argsort(dists)[:fan_out]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        for j in nearest_idx:
            ch2, h2, w2, mag2 = peaks[j]
            mag_bin2 = _quantize_magnitude(
                mag2, bin_edges,
            )
            dh = h2 - h1
            dw = w2 - w1
            h = (ch1
                 + ch2 * HASH_PRIME1
                 + (dh + 100) * HASH_PRIME1 * HASH_PRIME1
                 + (dw + 100) * 31
                 + abs(mag_bin1 - mag_bin2) * 97
                 ) % HASH_PRIME2
            hashes[h] += 1
    return hashes


# =====================================================================
def fingerprint_batch(model, x, device,
                      layer_names=None,
                      radius=DEFAULT_RADIUS,
                      top_k=DEFAULT_TOP_K,
                      fan_out=DEFAULT_FAN_OUT,
                      bin_edges=None):
    """Full fingerprint pipeline for an input batch.

    Parameters
    ----------
    model : torch.nn.Module
        The network with task_id set.
    x : torch.Tensor
        Input batch (B, C, H, W).
    device : torch.device
        Computation device.
    layer_names : list[str], default=None
        Layers to extract from.
    radius : int, default=3
        Peak picking radius.
    top_k : int, default=50
        Max peaks per sample per layer.
    fan_out : int, default=5
        Hashing fan-out.
    bin_edges : dict, default=None
        Mapping layer_name -> bin edges array. If None,
        bin edges are computed per sample.

    Returns
    -------
    batch_hashes : Counter
        Aggregated hash counts across all samples and
        layers.
    """
    if layer_names is None:
        layer_names = DEFAULT_LAYERS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    activations = extract_activation_maps(
        model, x, device, layer_names,
    )
    batch_hashes = Counter()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for layer_name, act_array in activations.items():
        layer_bin_edges = None
        if bin_edges is not None:
            layer_bin_edges = bin_edges.get(layer_name)
        for sample_idx in range(act_array.shape[0]):
            act = act_array[sample_idx]
            peaks = find_activation_peaks(
                act, radius=radius, top_k=top_k,
            )
            sample_hashes = hash_activation_peaks(
                peaks, fan_out=fan_out,
                bin_edges=layer_bin_edges,
            )
            batch_hashes.update(sample_hashes)
    return batch_hashes
