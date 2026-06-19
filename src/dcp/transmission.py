"""Transmission map estimation and refinement.

The haze imaging model is

    I(x) = J(x) t(x) + A (1 - t(x))

where I is the observed image, J the haze-free scene radiance, A the global
atmospheric light and t(x) in [0, 1] the medium transmission (fraction of
light reaching the camera without scattering).

Assuming the dehazed scene's dark channel -> 0 and A is known, He et al. derive
an estimate of the transmission directly from the dark channel of the
normalised image:

    t_raw(x) = 1 - omega * darkchannel( I(x) / A )

The constant omega (< 1) keeps a small amount of haze for distant objects so
the result still looks natural (aerial perspective).
"""

import cv2
import numpy as np

from .dark_channel import dark_channel
from .guided_filter import guided_filter


def estimate_transmission(
    image: np.ndarray, A: np.ndarray, omega: float = 0.95, patch_size: int = 15
) -> np.ndarray:
    """Estimate the raw (coarse) transmission map.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) float in [0, 1], BGR.
    A : np.ndarray
        Length-3 atmospheric light (BGR).
    omega : float
        Haze-retention factor (default 0.95).
    patch_size : int
        Patch size for the dark channel (default 15).

    Returns
    -------
    np.ndarray
        (H, W) raw transmission in [0, 1] (before guided-filter refinement).
    """
    normalized = image / A.reshape(1, 1, 3)
    transmission = 1.0 - omega * dark_channel(normalized, patch_size)
    return transmission


def refine_transmission(
    image: np.ndarray, transmission: np.ndarray, radius: int = 60, eps: float = 1e-3
) -> np.ndarray:
    """Refine a coarse transmission map with a guided filter.

    The grayscale version of the hazy input is used as the guidance image so
    the refined transmission follows real edges instead of patch blocks.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) float in [0, 1], BGR (used to build the grayscale guide).
    transmission : np.ndarray
        (H, W) coarse transmission map.
    radius, eps : guided-filter parameters.

    Returns
    -------
    np.ndarray
        (H, W) refined transmission map.
    """
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    return guided_filter(gray, transmission, radius, eps)
