#!/usr/bin/env python3
"""Run the SOTA learning-based dehazer (DehazeFormer) over an image or folder.

Mirrors run_dcp.py's output convention (``<stem>_dehazed.png``, input structure
mirrored under the output dir) so that metrics.py evaluates DCP and DehazeFormer
uniformly.

Examples
--------
    python src/run_sota.py --input data/real --output results/dehazeformer
    python src/run_sota.py --input data/synthetic/hazy --output results/dehazeformer
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from io_utils import list_images, read_image, write_image  # noqa: E402
from sota.dehazeformer_infer import DehazeFormerDehazer  # noqa: E402


def _rel_stem(path: str, input_root: str):
    base = os.path.dirname(input_root) if os.path.isfile(input_root) else input_root
    rel = os.path.relpath(path, base)
    return os.path.dirname(rel), os.path.splitext(os.path.basename(rel))[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="DehazeFormer (SOTA) dehazing")
    parser.add_argument("--input", required=True, help="input image or folder")
    parser.add_argument("--output", default="results/dehazeformer", help="output folder")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if unset)")
    args = parser.parse_args()

    paths = list_images(args.input)
    if not paths:
        sys.exit(f"no images found under {args.input!r}")

    dehazer = DehazeFormerDehazer(device=args.device)
    print(f"DehazeFormer on {dehazer.device}. {len(paths)} image(s) -> {args.output}")

    for i, path in enumerate(paths, 1):
        rel_dir, stem = _rel_stem(path, args.input)
        img = read_image(path)
        t0 = time.time()
        out = dehazer.dehaze(img)
        dt = time.time() - t0
        write_image(os.path.join(args.output, rel_dir, f"{stem}_dehazed.png"), out)
        print(f"[{i}/{len(paths)}] {os.path.basename(path):28s} "
              f"{img.shape[1]}x{img.shape[0]}  {dt*1000:.0f} ms")

    print("Done.")


if __name__ == "__main__":
    main()
