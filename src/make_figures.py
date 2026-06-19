#!/usr/bin/env python3
"""Generate all figures used in the report (saved to results/figures/).

Figures:
  1. pipeline.png            - the DCP pipeline on one image (input, dark channel,
                               raw & refined transmission, output).
  2. real_comparison.png     - real hazy images: hazy vs DCP vs DehazeFormer.
  3. synthetic_comparison.png- synthetic images: hazy vs DCP vs DehazeFormer vs GT.
  4. failure_bright.png      - DCP failure analysis on bright/white/sky regions,
                               using the intermediate maps as evidence.
  5. metrics_psnr_ssim.png   - PSNR/SSIM bar charts (DCP vs DehazeFormer) on the
                               synthetic set, with the hazy baseline marked.

All inputs are read as BGR (OpenCV) and converted to RGB for matplotlib.
"""

from __future__ import annotations

import csv
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dcp import dehaze  # noqa: E402
from io_utils import read_image  # noqa: E402

FIG_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def rgb_arr(img_bgr_float: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(img_bgr_float, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# 1. DCP pipeline
# --------------------------------------------------------------------------- #
def fig_pipeline(image_path: str):
    img = read_image(image_path)
    out, inter = dehaze(img, return_intermediate=True)

    panels = [
        (rgb_arr(img), "Hazy input  I", None),
        (inter["dark_channel"], "Dark channel  $J^{dark}$", "gray"),
        (inter["transmission_raw"], "Raw transmission  $\\tilde{t}$", "viridis"),
        (inter["transmission_refined"], "Refined transmission  $t$ (guided filter)", "viridis"),
        (rgb_arr(out), "Dehazed output  J", None),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.2))
    for ax, (data, title, cmap) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        if cmap:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    A = inter["atmospheric_light"][::-1]  # BGR->RGB for display
    fig.suptitle(f"Dark Channel Prior pipeline   (estimated atmospheric light A = "
                 f"[{A[0]:.2f}, {A[1]:.2f}, {A[2]:.2f}] RGB)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(FIG_DIR, "pipeline.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


# --------------------------------------------------------------------------- #
# 2 & 3. comparison grids
# --------------------------------------------------------------------------- #
def _grid(rows, col_titles, out_name, suptitle):
    n_rows, n_cols = len(rows), len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    for r, (label, imgs) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(imgs[c])
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12)
        axes[r, 0].set_ylabel(label, fontsize=10)
        # ylabel needs axis on
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(FIG_DIR, out_name)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def fig_real_comparison():
    specs = [
        ("tiananmen1", "cityscape", "real"),
        ("nyc_foggy_skyline", "cityscape", "real"),
        ("img_8766", "cityscape", "real"),
        ("mountain", "landscape", "real"),
        ("forest1", "landscape", "real"),
        ("tree2", "landscape", "real"),
    ]
    rows = []
    for stem, cat, split in specs:
        hazy = rgb(_first(f"data/real/{cat}/{stem}.*"))
        dcp = rgb(f"{ROOT}/results/dcp/{split}/{cat}/{stem}_dehazed.png")
        df = rgb(f"{ROOT}/results/dehazeformer/{split}/{cat}/{stem}_dehazed.png")
        rows.append((stem, [hazy, dcp, df]))
    _grid(rows, ["Hazy input", "DCP (ours)", "DehazeFormer"],
          "real_comparison.png", "Real hazy images: DCP vs DehazeFormer")


def fig_synthetic_comparison():
    specs = [
        ("kodim08", "scene", "medium"),
        ("kodim24", "landscape", "medium"),
        ("banquet_room", "indoor", "medium"),
        ("kodim21_lighthouse", "bright", "dense"),
        ("everest_snow", "bright", "dense"),
    ]
    rows = []
    for stem, cat, lvl in specs:
        hazy = rgb(f"{ROOT}/data/synthetic/hazy/{cat}/{stem}_{lvl}.png")
        dcp = rgb(f"{ROOT}/results/dcp/synthetic/{cat}/{stem}_{lvl}_dehazed.png")
        df = rgb(f"{ROOT}/results/dehazeformer/synthetic/{cat}/{stem}_{lvl}_dehazed.png")
        gt = rgb(f"{ROOT}/data/synthetic/gt/{cat}/{stem}.png")
        rows.append((f"{stem}\n({lvl})", [hazy, dcp, df, gt]))
    _grid(rows, ["Hazy input", "DCP (ours)", "DehazeFormer", "Ground truth"],
          "synthetic_comparison.png",
          "Synthetic hazy images with ground truth: DCP vs DehazeFormer")


# --------------------------------------------------------------------------- #
# 4. failure analysis on bright/white/sky regions
# --------------------------------------------------------------------------- #
def fig_failure():
    cases = [
        ("data/synthetic/hazy/bright/everest_snow_dense.png",
         "Synthetic: snow + sky (white scene)"),
        ("data/real/landscape/mountain.png",
         "Real: large sky region"),
    ]
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 4.3 * len(cases)))
    for r, (path, label) in enumerate(cases):
        img = read_image(os.path.join(ROOT, path))
        out, inter = dehaze(img, return_intermediate=True)
        dark = inter["dark_channel"]
        t = inter["transmission_refined"]
        panels = [
            (rgb_arr(img), "Hazy input", None),
            (dark, "Dark channel (bright = prior violated)", "inferno"),
            (t, "Transmission t (low = over-dehazed)", "viridis"),
            (rgb_arr(out), "DCP output (artefacts in bright areas)", None),
        ]
        for c, (data, title, cmap) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=11)
            if cmap:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[r, 0].set_title(f"{label}\n{panels[0][1]}" if r else panels[0][1], fontsize=11)
        axes[r, 0].text(-0.04, 0.5, label, rotation=90, va="center", ha="right",
                        transform=axes[r, 0].transAxes, fontsize=11, fontweight="bold")
    fig.suptitle("Why DCP fails on bright / white / sky regions: the dark channel is NOT near zero there,\n"
                 "so transmission is under-estimated and those regions are over-dehazed (colour shift, darkening, noise).",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(FIG_DIR, "failure_bright.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


# --------------------------------------------------------------------------- #
# 5. metrics bar charts
# --------------------------------------------------------------------------- #
def fig_metrics():
    rows = _read_metrics()
    fr = [r for r in rows if r["psnr"] not in ("", None)]
    # build per-image dict keyed by (image) -> {method: (psnr,ssim)}, plus hazy baseline
    keys = sorted({r["image"] for r in fr})
    methods = ["dcp", "dehazeformer"]
    psnr = {m: [] for m in methods}
    ssim = {m: [] for m in methods}
    psnr_hz, ssim_hz = [], []
    for k in keys:
        for m in methods:
            match = next((r for r in fr if r["image"] == k and r["method"] == m), None)
            psnr[m].append(float(match["psnr"]) if match else 0)
            ssim[m].append(float(match["ssim"]) if match else 0)
        base = next((r for r in fr if r["image"] == k and r.get("psnr_hazy")), None)
        psnr_hz.append(float(base["psnr_hazy"]) if base and base.get("psnr_hazy") else 0)
        ssim_hz.append(float(base["ssim_hazy"]) if base and base.get("ssim_hazy") else 0)

    x = np.arange(len(keys))
    w = 0.38
    labels = [k.replace("_", "\n", 1) for k in keys]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.5))

    a1.bar(x - w / 2, psnr["dcp"], w, label="DCP", color="#4C72B0")
    a1.bar(x + w / 2, psnr["dehazeformer"], w, label="DehazeFormer", color="#DD8452")
    a1.plot(x, psnr_hz, "k_", markersize=18, label="Hazy (no dehazing)")
    a1.set_ylabel("PSNR (dB)  - higher is better"); a1.set_title("PSNR on synthetic set")
    a1.set_xticks(x); a1.set_xticklabels(labels, fontsize=8); a1.legend(); a1.grid(axis="y", alpha=0.3)

    a2.bar(x - w / 2, ssim["dcp"], w, label="DCP", color="#4C72B0")
    a2.bar(x + w / 2, ssim["dehazeformer"], w, label="DehazeFormer", color="#DD8452")
    a2.plot(x, ssim_hz, "k_", markersize=18, label="Hazy (no dehazing)")
    a2.set_ylabel("SSIM  - higher is better"); a2.set_title("SSIM on synthetic set")
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=8); a2.legend(); a2.grid(axis="y", alpha=0.3)

    fig.suptitle("Quantitative comparison (full-reference, synthetic ground truth)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(FIG_DIR, "metrics_psnr_ssim.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


# --------------------------------------------------------------------------- #
def _first(pattern: str) -> str:
    import glob
    hits = glob.glob(os.path.join(ROOT, pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def _read_metrics():
    path = os.path.join(ROOT, "results", "metrics.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    fig_pipeline(os.path.join(ROOT, "data/real/landscape/mountain.png"))
    fig_real_comparison()
    fig_synthetic_comparison()
    fig_failure()
    fig_metrics()
    print("\nAll figures written to", FIG_DIR)


if __name__ == "__main__":
    main()
