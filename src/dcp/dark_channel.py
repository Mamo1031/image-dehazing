"""Dark channel computation.

The dark channel of an image J is

    J_dark(x) = min_{c in {r,g,b}} ( min_{y in Omega(x)} J^c(y) )

i.e. the per-pixel minimum across colour channels, followed by a local
minimum (morphological erosion) over a square patch Omega of side `patch_size`.

Key empirical prior (He et al., 2009): for haze-free outdoor images, the dark
channel is close to zero everywhere except in sky/bright regions, because most
local patches contain at least one very dark pixel in some colour channel.
"""

import cv2
import numpy as np


def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Compute the dark channel of an image.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) float array in [0, 1]. Channel order is irrelevant here
        because we take the minimum across channels.
    patch_size : int
        Side length of the square patch Omega (odd number, default 15).

    Returns
    -------
    np.ndarray
        (H, W) float array: the dark channel.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got shape {image.shape}")
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    # Erosion == local minimum over the structuring element.
    dark = cv2.erode(min_channel, kernel)
    return dark
