"""End-to-end Dark Channel Prior dehazing pipeline."""

from __future__ import annotations

import numpy as np

from .atmospheric_light import estimate_atmospheric_light
from .dark_channel import dark_channel
from .transmission import estimate_transmission, refine_transmission


def dehaze(
    image: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    top_percent: float = 0.001,
    gf_radius: int = 60,
    gf_eps: float = 1e-3,
    atm_mode: str = "mean",
    return_intermediate: bool = False,
):
    """Remove haze from a single image using the Dark Channel Prior.

    Pipeline: dark channel -> atmospheric light A -> raw transmission ->
    guided-filter refinement -> recover scene radiance

        J(x) = (I(x) - A) / max(t(x), t0) + A

    The lower bound t0 prevents division by near-zero transmission (which would
    otherwise produce extreme noise/colour in the densest-haze regions).

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3) float in [0, 1], BGR.
    patch_size, omega, t0, top_percent, gf_radius, gf_eps, atm_mode :
        Algorithm hyper-parameters (see the individual modules).
    return_intermediate : bool
        If True, also return a dict of intermediate maps for visualisation.

    Returns
    -------
    np.ndarray | tuple[np.ndarray, dict]
        Dehazed image (float [0, 1], BGR); plus an intermediates dict when
        `return_intermediate` is True.
    """
    image = image.astype(np.float32)

    dark = dark_channel(image, patch_size)
    A = estimate_atmospheric_light(image, dark, top_percent=top_percent, mode=atm_mode)
    t_raw = estimate_transmission(image, A, omega, patch_size)
    t_refined = refine_transmission(image, t_raw, gf_radius, gf_eps)

    t_clamped = np.clip(t_refined, t0, 1.0)
    radiance = (image - A.reshape(1, 1, 3)) / t_clamped[..., None] + A.reshape(1, 1, 3)
    radiance = np.clip(radiance, 0.0, 1.0)

    if return_intermediate:
        intermediates = {
            "dark_channel": dark,
            "atmospheric_light": A,
            "transmission_raw": t_raw,
            "transmission_refined": t_refined,
        }
        return radiance, intermediates
    return radiance
