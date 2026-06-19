"""Small shared image I/O helpers (used by run_dcp / run_sota / metrics / figures).

Convention: in-memory images are float32 in [0, 1], BGR channel order (OpenCV).
"""

from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def read_image(path: str) -> np.ndarray:
    """Read an image as float32 [0, 1], BGR, shape (H, W, 3)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img.astype(np.float32) / 255.0


def write_image(path: str, img: np.ndarray) -> None:
    """Write a float [0, 1] BGR image to disk (created dirs as needed)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = np.clip(img, 0.0, 1.0)
    cv2.imwrite(path, (out * 255.0 + 0.5).astype(np.uint8))


def write_gray(path: str, img: np.ndarray) -> None:
    """Write a single-channel float map (e.g. transmission) as an 8-bit image."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = np.clip(img, 0.0, 1.0)
    cv2.imwrite(path, (out * 255.0 + 0.5).astype(np.uint8))


def list_images(root: str, recursive: bool = True) -> List[str]:
    """Return sorted image paths under `root` (or just `root` if it is a file)."""
    if os.path.isfile(root):
        return [root]
    paths: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(IMAGE_EXTS):
                    paths.append(os.path.join(dirpath, name))
    else:
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if os.path.isfile(full) and name.lower().endswith(IMAGE_EXTS):
                paths.append(full)
    return sorted(paths)
