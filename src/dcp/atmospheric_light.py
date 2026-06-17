"""Atmospheric light estimation.

He et al. estimate the global atmospheric light A from the haziest pixels.
The dark channel is brightest where haze is densest, so we:

  1. take the top `top_percent` brightest pixels of the dark channel, then
  2. among those candidate pixels, read their colours from the input image.

The paper picks the single pixel with the highest intensity. In practice
averaging the candidate pixels (as in the popular He-Zhang reference
implementation) is more robust to outliers, so that is the default here.
This is precisely the step that mis-fires on images with large white/bright
regions: white objects also produce a bright dark channel and get mistaken
for "haziest", inflating A (discussed in the report's failure analysis).
"""

from __future__ import annotations

import numpy as np

from .dark_channel import dark_channel


def estimate_atmospheric_light(
    image: np.ndarray,
    dark: np.ndarray | None = None,
    patch_size: int = 15,
    top_percent: float = 0.001,
    mode: str = "mean",
) -> np.ndarray:
    """Estimate the atmospheric light A (a length-3 vector).

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) float array in [0, 1], BGR.
    dark : np.ndarray, optional
        Pre-computed dark channel; recomputed if None.
    patch_size : int
        Patch size used if `dark` must be computed.
    top_percent : float
        Fraction of brightest dark-channel pixels to consider (default 0.1%).
    mode : {"mean", "max"}
        "mean": average the candidate pixels (robust, default).
        "max":  pick the single brightest-intensity candidate (paper-faithful).

    Returns
    -------
    np.ndarray
        Length-3 atmospheric light vector (BGR), values in [0, 1].
    """
    if dark is None:
        dark = dark_channel(image, patch_size)

    h, w = dark.shape
    n_pixels = h * w
    n_top = max(int(n_pixels * top_percent), 1)

    dark_flat = dark.reshape(n_pixels)
    image_flat = image.reshape(n_pixels, 3)

    # Indices of the `n_top` brightest dark-channel pixels.
    candidate_idx = np.argpartition(dark_flat, n_pixels - n_top)[n_pixels - n_top:]
    candidates = image_flat[candidate_idx]

    if mode == "max":
        A = candidates[np.argmax(candidates.sum(axis=1))]
    elif mode == "mean":
        A = candidates.mean(axis=0)
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'mean' or 'max'")

    # Guard against degenerate near-zero values that would blow up division.
    return np.maximum(A, 1e-3)
