# Image Dehazing Toolkit — Dark Channel Prior & DehazeFormer

A small, self-contained toolkit for **single image dehazing**. It provides a
from-scratch implementation of the **Dark Channel Prior** (DCP, He et al.,
CVPR 2009) and an inference wrapper around a modern learning-based method,
**DehazeFormer** (a Vision Transformer, IEEE TIP 2023), plus quantitative
metrics and figure helpers. Run either method on your own images from the
command line, or import the DCP pipeline as a library.

## Install

The project is managed with [**uv**](https://docs.astral.sh/uv/); dependencies
are pinned in `pyproject.toml` / `uv.lock`.

```bash
uv sync                      # core: DCP + metrics (NumPy/OpenCV only, no GPU needed)
uv sync --extra dehazeformer # also install PyTorch for the DehazeFormer method
```

The DCP pipeline needs no GPU and no PyTorch. PyTorch (pinned to the CUDA 11.6
build) is an optional extra, only required for DehazeFormer.

## Usage

Run DCP over a single image or a whole folder (the input directory structure is
mirrored under `--output`):

```bash
uv run src/run_dcp.py --input path/to/image_or_folder --output out/
uv run src/run_dcp.py --input img.jpg --output out/ --save-intermediate  # + dark channel / transmission maps
```

`run_dcp.py` exposes the algorithm's hyper-parameters as flags
(`--patch-size`, `--omega`, `--t0`, `--gf-radius`, …). Run the learning-based
method (first fetch its weights with `bash scripts/setup_sota.sh`):

```bash
uv run --extra dehazeformer src/run_sota.py --input path/to/folder --output out/   # DehazeFormer
```

As a library:

```python
from dcp import dehaze            # src/ is the import root
clear = dehaze(img)               # img: (H, W, 3) float in [0, 1], BGR
clear, maps = dehaze(img, return_intermediate=True)  # also dark channel + transmission
```

## What it does

- **DCP from scratch** (`src/dcp/`): dark channel, atmospheric-light estimation,
  transmission estimation, guided-filter refinement, scene-radiance recovery.
  Validated against the reference implementation (`forest1` output matches to 3 d.p.).
- **SOTA wrapper** (`src/sota/`): DehazeFormer (the authors' mixed-dataset MCT demo
  model, fetched from HuggingFace), resolution-independent and fast.
- **Metrics** (`src/metrics.py`): full-reference PSNR/SSIM + no-reference entropy,
  contrast, average gradient and Hautière's newly-visible-edges ratio.
- **Figures** (`src/make_figures.py`): comparison/pipeline/failure-analysis plots.

## Layout

```
src/dcp/          DCP implementation (modular, dependency-free of torch)
src/sota/         DehazeFormer inference wrapper
src/run_dcp.py    CLI: DCP over an image/folder (+ intermediate maps)
src/run_sota.py   CLI: DehazeFormer over an image/folder
src/io_utils.py   image read/write helpers
src/metrics.py    quantitative evaluation -> results/metrics.csv
src/make_figures.py  figure helpers (comparison / pipeline / failure plots)
scripts/          download_data.py, make_synthetic.py, setup_sota.sh
data/real/        sample hazy images (data/synthetic/ + data/clean/ are generated, git-ignored)
results/          default output dir (generated outputs, git-ignored)
pyproject.toml    uv project + pinned deps (torch is an optional extra); uv.lock is the lockfile
```

## DCP vs DehazeFormer

- DCP is a strong, **training-free** baseline (~+11 dB PSNR on synthetic haze) but
  **over-dehazes bright/white/sky regions** because the dark-channel prior is violated
  there.
- DehazeFormer is **more natural on real images** and **more accurate on medium haze**
  (+2.6 dB), but is conservative on out-of-distribution **dense** haze (where DCP, the
  matched inverse of the synthetic haze model, wins by +4.5 dB).

## Credits / references

He et al., *Single Image Haze Removal Using Dark Channel Prior*, CVPR 2009 ·
He et al., *Guided Image Filtering*, ECCV 2010 ·
Song et al., *Vision Transformers for Single Image Dehazing*, IEEE TIP 2023
([code](https://github.com/IDKiro/DehazeFormer)) ·
reference DCP code: <https://github.com/He-Zhang/image_dehaze>.
Test images: He-Zhang & Color-Attenuation repos, Kodak set, Wikimedia Commons.
