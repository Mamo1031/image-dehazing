#!/usr/bin/env python3
"""Synthesize hazy/clean image pairs with known ground truth.

Applies the atmospheric scattering model used throughout the report

    I(x) = J(x) * t(x) + A * (1 - t(x)),   t(x) = exp(-beta * d(x))

to the clean images in ``data/clean/`` to produce ground-truth pairs in
``data/synthetic/{hazy,gt}/``. Because the clean image J is known, this enables
full-reference PSNR/SSIM evaluation of any dehazing method.

Depth proxy d(x): a normalized vertical gradient (haze grows denser toward the
top / horizon) blended with low-frequency noise so the haze is not a flat band.
This is an approximation of true scene depth (RESIDE uses NYU depth maps); it is
sufficient and transparent for a controlled quantitative experiment. The bright
cases (snow, sky) deliberately exercise the Dark Channel Prior's failure mode.

Reproducible: fixed RNG seed; deterministic atmospheric light per image.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from io_utils import list_images, read_image, write_image  # noqa: E402

# beta per haze level (t_min = exp(-beta) at the farthest depth d=1)
LEVELS = {"medium": 1.2, "dense": 2.5}
# only generate the heavy "dense" level for the bright failure cases
DENSE_ONLY_CATEGORIES = {"bright"}


def depth_proxy(h: int, w: int, seed: int) -> np.ndarray:
    """Smooth depth map in [0, 1]: vertical gradient + low-frequency noise."""
    rng = np.random.default_rng(seed)
    vert = np.linspace(1.0, 0.0, h)[:, None] * np.ones((1, w))  # 1=top(far) .. 0=bottom(near)
    noise = rng.random((max(h // 32, 2), max(w // 32, 2))).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(w, h) / 40.0)
    d = 0.75 * vert + 0.25 * noise
    d -= d.min()
    d /= (d.max() + 1e-8)
    return d.astype(np.float32)


def atmospheric_light(seed: int) -> np.ndarray:
    """Deterministic bright, near-neutral atmospheric light (BGR) in [0.85, 0.97]."""
    rng = np.random.default_rng(seed + 999)
    return (0.88 + 0.08 * rng.random(3)).astype(np.float32)


def main() -> None:
    clean_paths = list_images(os.path.join(ROOT, "data", "clean"))
    if not clean_paths:
        sys.exit("no clean images found under data/clean — run download_data.py first")

    made = 0
    for seed, path in enumerate(clean_paths):
        category = os.path.basename(os.path.dirname(path))
        name = os.path.splitext(os.path.basename(path))[0]
        J = read_image(path)
        h, w = J.shape[:2]
        d = depth_proxy(h, w, seed)
        A = atmospheric_light(seed)

        gt_path = os.path.join(ROOT, "data", "synthetic", "gt", category, f"{name}.png")
        write_image(gt_path, J)

        levels = ["dense"] if category in DENSE_ONLY_CATEGORIES else ["medium", "dense"]
        for level in levels:
            beta = LEVELS[level]
            t = np.exp(-beta * d)[..., None]
            hazy = J * t + A.reshape(1, 1, 3) * (1.0 - t)
            out = os.path.join(ROOT, "data", "synthetic", "hazy", category,
                               f"{name}_{level}.png")
            write_image(out, hazy)
            made += 1
            print(f"  {category:9s} {name:22s} {level:6s} "
                  f"A(BGR)=[{A[0]:.2f},{A[1]:.2f},{A[2]:.2f}] beta={beta} -> {w}x{h}")

    print(f"\nGenerated {made} hazy image(s) with ground truth under data/synthetic/.")


if __name__ == "__main__":
    main()
