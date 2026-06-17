"""DehazeFormer inference wrapper (the modern / SOTA baseline).

Uses the official authors' demo model (IDKiro/DehazeFormer_Demo): the MCT
variant of DehazeFormer trained on a *mixed* dataset, which the authors
recommend for real-world hazy images. The heavy Transformer backbone runs on a
256x256 down-sampled copy to predict per-pixel colour curves, which are then
applied at full resolution via grid_sample - so any input size is handled
cheaply (Song et al., "Vision Transformers for Single Image Dehazing",
IEEE TIP 2023; MCT: https://github.com/IDKiro/MCT).

Model code + weights live under external/DehazeFormer_Demo/ (fetched by the
setup step). Input convention: RGB, scaled to [-1, 1]; output clamped to
[-1, 1] then mapped to [0, 1].
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import torch


def _get_coord_device_safe(self, x):
    """Device-safe replacement for MCT.get_coord.

    The upstream demo builds the coordinate grid on the CPU (it only ever ran on
    CPU), which crashes when the model is on CUDA. This version creates the grid
    on the input tensor's device.
    """
    B, _, H, W = x.size()
    coordh, coordw = torch.meshgrid(
        [torch.linspace(-1, 1, H, device=x.device),
         torch.linspace(-1, 1, W, device=x.device)], indexing="ij")
    coordh = coordh.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1, 1)
    coordw = coordw.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1, 1)
    return coordw.detach(), coordh.detach()

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_DIR = os.path.join(ROOT, "external", "DehazeFormer_Demo")
DEFAULT_WEIGHTS = os.path.join(DEMO_DIR, "saved_models", "dehazeformer.pth")


class DehazeFormerDehazer:
    """Thin wrapper: construct once, then call ``dehaze(img)`` per image."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS, device: str | None = None):
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"DehazeFormer weights not found at {weights}. "
                "Run scripts/setup_sota.sh (or re-download the demo model).")
        if not os.path.isdir(DEMO_DIR):
            raise FileNotFoundError(f"DehazeFormer demo code not found at {DEMO_DIR}.")

        # Import the demo's model package (defines dehazeformer == MCT).
        if DEMO_DIR not in sys.path:
            sys.path.insert(0, DEMO_DIR)
        from models import dehazeformer  # noqa: E402  (external package)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = dehazeformer()
        state = torch.load(weights, map_location="cpu")["state_dict"]
        self.net.load_state_dict(state)
        # Bind the device-safe coordinate grid (see helper above).
        self.net.get_coord = types.MethodType(_get_coord_device_safe, self.net)
        self.net.eval().to(self.device)

    @torch.no_grad()
    def dehaze(self, image: np.ndarray) -> np.ndarray:
        """Dehaze a float [0, 1] BGR image; returns float [0, 1] BGR."""
        rgb = np.ascontiguousarray(image[:, :, ::-1])               # BGR -> RGB
        x = torch.from_numpy((rgb * 2.0 - 1.0).transpose(2, 0, 1))  # CHW, [-1,1]
        x = x.unsqueeze(0).float().to(self.device)
        out = self.net(x).clamp_(-1.0, 1.0)[0] * 0.5 + 0.5          # [0,1]
        out = out.permute(1, 2, 0).cpu().numpy()                    # HWC, RGB
        return np.ascontiguousarray(out[:, :, ::-1])                # RGB -> BGR
