#!/usr/bin/env python3
"""Quantitative evaluation of dehazing results.

Full-reference (needs ground truth, i.e. the synthetic set):
  * PSNR  - peak signal-to-noise ratio (dB), higher is better
  * SSIM  - structural similarity, in [0, 1], higher is better

No-reference (works on real images too; compares output vs hazy input):
  * entropy        - Shannon entropy of the luminance (more detail -> higher)
  * contrast       - std of luminance
  * avg_gradient   - mean gradient magnitude (sharpness)
  * e              - Hautiere's rate of newly visible edges, (n_out-n_in)/n_in

CLI: walks results/<method>/<split>/<category>/<stem>_dehazed.png, matches each
to its hazy input (and ground truth for the synthetic split), and writes
results/metrics.csv plus a printed summary.
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from io_utils import read_image  # noqa: E402

HAZE_LEVELS = ("light", "medium", "dense")


# --------------------------------------------------------------------------- #
# metric primitives (inputs are float [0, 1], BGR, shape HxWx3)
# --------------------------------------------------------------------------- #
def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _match(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    if test.shape != ref.shape:
        test = cv2.resize(test, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    return test


def psnr(ref: np.ndarray, test: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, _match(ref, test), data_range=1.0))


def ssim(ref: np.ndarray, test: np.ndarray) -> float:
    return float(structural_similarity(ref, _match(ref, test), channel_axis=-1, data_range=1.0))


def entropy(img: np.ndarray) -> float:
    g = (_gray(img) * 255).astype(np.uint8)
    hist = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def contrast(img: np.ndarray) -> float:
    return float(_gray(img).std())


def avg_gradient(img: np.ndarray) -> float:
    g = _gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def new_visible_edges(hazy: np.ndarray, out: np.ndarray, thresh: float = 0.03) -> float:
    """Hautiere e: relative increase in number of visible (high-gradient) edges."""
    def n_edges(img):
        g = _gray(img)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return int((np.sqrt(gx ** 2 + gy ** 2) > thresh).sum())
    n_in = n_edges(hazy)
    n_out = n_edges(_match(hazy, out))
    return float((n_out - n_in) / max(n_in, 1))


# --------------------------------------------------------------------------- #
# results <-> inputs/GT mapping
# --------------------------------------------------------------------------- #
def _find_hazy(split: str, category: str, name: str) -> Optional[str]:
    base = "data/real" if split == "real" else "data/synthetic/hazy"
    hits = glob.glob(os.path.join(ROOT, base, category, name + ".*"))
    return hits[0] if hits else None


def _find_gt(category: str, name: str) -> Optional[str]:
    gt_name = name
    for lvl in HAZE_LEVELS:
        if gt_name.endswith("_" + lvl):
            gt_name = gt_name[: -(len(lvl) + 1)]
            break
    hits = glob.glob(os.path.join(ROOT, "data/synthetic/gt", category, gt_name + ".*"))
    return hits[0] if hits else None


def _level_of(name: str) -> str:
    for lvl in HAZE_LEVELS:
        if name.endswith("_" + lvl):
            return lvl
    return ""


def main() -> None:
    method_dirs = [d for d in glob.glob(os.path.join(ROOT, "results", "*"))
                   if os.path.isdir(d) and os.path.basename(d) != "figures"]
    rows = []
    for mdir in sorted(method_dirs):
        method = os.path.basename(mdir)
        for res_path in sorted(glob.glob(os.path.join(mdir, "**", "*_dehazed.png"), recursive=True)):
            rel = os.path.relpath(res_path, mdir)              # split/category/stem_dehazed.png
            parts = rel.split(os.sep)
            if len(parts) < 3:
                continue
            split, category = parts[0], parts[1]
            name = os.path.basename(res_path)[: -len("_dehazed.png")]
            hazy_path = _find_hazy(split, category, name)
            if hazy_path is None:
                print(f"  ! no hazy input for {res_path}", file=sys.stderr)
                continue
            hazy = read_image(hazy_path)
            out = read_image(res_path)

            row = {
                "method": method, "split": split, "category": category,
                "image": name, "level": _level_of(name),
                "psnr": "", "ssim": "",
                "entropy_in": round(entropy(hazy), 3), "entropy_out": round(entropy(out), 3),
                "contrast_in": round(contrast(hazy), 4), "contrast_out": round(contrast(out), 4),
                "avg_grad_in": round(avg_gradient(hazy), 4), "avg_grad_out": round(avg_gradient(out), 4),
                "e_new_edges": round(new_visible_edges(hazy, out), 3),
            }
            gt_path = _find_gt(category, name) if split == "synthetic" else None
            if gt_path:
                gt = read_image(gt_path)
                row["psnr"] = round(psnr(gt, out), 2)
                row["ssim"] = round(ssim(gt, out), 4)
                # also the hazy baseline PSNR/SSIM for reference (how bad before dehazing)
                row["psnr_hazy"] = round(psnr(gt, hazy), 2)
                row["ssim_hazy"] = round(ssim(gt, hazy), 4)
            rows.append(row)

    if not rows:
        sys.exit("no dehazed results found under results/ — run run_dcp.py first")

    # union of keys, stable column order
    cols = ["method", "split", "category", "image", "level",
            "psnr", "ssim", "psnr_hazy", "ssim_hazy",
            "entropy_in", "entropy_out", "contrast_in", "contrast_out",
            "avg_grad_in", "avg_grad_out", "e_new_edges"]
    out_csv = os.path.join(ROOT, "results", "metrics.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    print(f"Wrote {out_csv}  ({len(rows)} rows)\n")
    # printed summary: full-reference rows
    fr = [r for r in rows if r["psnr"] != ""]
    if fr:
        print("Full-reference (synthetic, with GT):")
        print(f"  {'method':14s}{'image':27s}{'level':8s}{'PSNR':>7s}{'SSIM':>8s}"
              f"{'PSNR_hz':>9s}{'SSIM_hz':>9s}")
        for r in sorted(fr, key=lambda x: (x["method"], x["category"], x["image"])):
            print(f"  {r['method']:14s}{r['image']:27s}{r['level']:8s}"
                  f"{r['psnr']:>7.2f}{r['ssim']:>8.4f}{str(r.get('psnr_hazy','')):>9}{str(r.get('ssim_hazy','')):>9}")
    print("\nNo-reference (all):  see results/metrics.csv "
          "(entropy/contrast/avg_grad in vs out, e=new visible edges).")


if __name__ == "__main__":
    main()
