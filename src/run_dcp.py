#!/usr/bin/env python3
"""Run Dark Channel Prior dehazing over an image or a folder of images.

Saves, for every input image, the dehazed result and (optionally) the
intermediate maps used by the algorithm: dark channel, raw transmission and
guided-filter-refined transmission. The directory structure of the input is
mirrored under the output directory.

Examples
--------
    python src/run_dcp.py --input data --output results/dcp --save-intermediate
    python src/run_dcp.py --input data/bright/snow.jpg --output results/dcp
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Make `import dcp` work regardless of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dcp import dehaze  # noqa: E402
from io_utils import list_images, read_image, write_gray, write_image  # noqa: E402


def _rel_stem(path: str, input_root: str):
    """Return (relative_dir, stem) of `path` relative to `input_root`."""
    if os.path.isfile(input_root):
        base = os.path.dirname(input_root)
    else:
        base = input_root
    rel = os.path.relpath(path, base)
    rel_dir = os.path.dirname(rel)
    stem = os.path.splitext(os.path.basename(rel))[0]
    return rel_dir, stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Dark Channel Prior dehazing")
    parser.add_argument("--input", required=True, help="input image or folder")
    parser.add_argument("--output", default="results/dcp", help="output folder")
    parser.add_argument("--save-intermediate", action="store_true",
                        help="also save dark channel + transmission maps")
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--omega", type=float, default=0.95)
    parser.add_argument("--t0", type=float, default=0.1)
    parser.add_argument("--top-percent", type=float, default=0.001)
    parser.add_argument("--gf-radius", type=int, default=60)
    parser.add_argument("--gf-eps", type=float, default=1e-3)
    parser.add_argument("--atm-mode", choices=["mean", "max"], default="mean")
    args = parser.parse_args()

    paths = list_images(args.input)
    if not paths:
        print(f"no images found under {args.input!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} image(s). Output -> {args.output}")
    inter_root = os.path.join(args.output, "intermediate")

    for i, path in enumerate(paths, 1):
        rel_dir, stem = _rel_stem(path, args.input)
        img = read_image(path)

        t_start = time.time()
        result, inter = dehaze(
            img,
            patch_size=args.patch_size,
            omega=args.omega,
            t0=args.t0,
            top_percent=args.top_percent,
            gf_radius=args.gf_radius,
            gf_eps=args.gf_eps,
            atm_mode=args.atm_mode,
            return_intermediate=True,
        )
        elapsed = time.time() - t_start

        out_path = os.path.join(args.output, rel_dir, f"{stem}_dehazed.png")
        write_image(out_path, result)

        if args.save_intermediate:
            d = os.path.join(inter_root, rel_dir)
            write_gray(os.path.join(d, f"{stem}_dark.png"), inter["dark_channel"])
            write_gray(os.path.join(d, f"{stem}_trans_raw.png"), inter["transmission_raw"])
            write_gray(os.path.join(d, f"{stem}_trans_refined.png"), inter["transmission_refined"])

        A = inter["atmospheric_light"]
        print(f"[{i}/{len(paths)}] {os.path.basename(path):28s} "
              f"A(BGR)=[{A[0]:.3f},{A[1]:.3f},{A[2]:.3f}]  {img.shape[1]}x{img.shape[0]}  "
              f"{elapsed*1000:.0f} ms")

    print("Done.")


if __name__ == "__main__":
    main()
