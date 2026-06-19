"""Guided image filtering (He, Sun, Tang).

An edge-preserving smoothing filter. We use it to refine the coarse,
patch-based transmission map so that its edges align with the real object
boundaries in the image, removing the block artefacts produced by the
patch-wise dark channel. It is the fast, modern replacement for the soft
matting used in the original 2009 paper.

The output q is a locally-linear function of the guidance image I:

    q_i = a_k * I_i + b_k,   for i in window k

with (a_k, b_k) chosen to be closest to the filtering input p in a
least-squares sense with regularisation eps. Box filters make it O(N).
"""

import cv2
import numpy as np


def _box(img: np.ndarray, radius: int) -> np.ndarray:
    """Box (mean) filter with window side 2*radius+1."""
    ksize = 2 * radius + 1
    return cv2.blur(img, (ksize, ksize))


def guided_filter(
    guide: np.ndarray, src: np.ndarray, radius: int = 60, eps: float = 1e-3
) -> np.ndarray:
    """Apply a guided filter.

    Parameters
    ----------
    guide : np.ndarray
        (H, W) single-channel guidance image, float in [0, 1].
    src : np.ndarray
        (H, W) input to be filtered (e.g. the raw transmission map), float.
    radius : int
        Window radius (default 60).
    eps : float
        Regularisation; larger -> smoother (default 1e-3).

    Returns
    -------
    np.ndarray
        (H, W) filtered output, same dtype-domain as `src`.
    """
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)

    mean_I = _box(guide, radius)
    mean_p = _box(src, radius)
    mean_Ip = _box(guide * src, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _box(guide * guide, radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box(a, radius)
    mean_b = _box(b, radius)

    return mean_a * guide + mean_b
