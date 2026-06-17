#!/usr/bin/env python3
"""Download test images for the dehazing exercise.

Two groups are fetched:

1. REAL hazy images (for qualitative analysis + no-reference metrics), saved under
   ``data/real/<category>/``. Sources: the He-Zhang DCP reference repo, the
   Color-Attenuation-Prior repo, and Wikimedia Commons.

2. CLEAN source images (for the synthetic ground-truth experiment), saved under
   ``data/clean/<category>/``. ``scripts/make_synthetic.py`` turns these into
   hazy/clean pairs so that PSNR/SSIM can be measured. Sources: the Kodak image
   set and Wikimedia Commons.

Every download is wrapped in try/except so a single dead link cannot break the
run; a summary of what actually succeeded is printed at the end.
"""

from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "image-dehazing-exercise/1.0 (educational use)"}
MAX_DIM = 1024  # cap the long side so DCP / DehazeFormer stay fast


# ---- direct URLs: (category, filename, url) --------------------------------
REAL_DIRECT = [
    ("landscape", "forest1.jpg",
     "https://raw.githubusercontent.com/He-Zhang/image_dehaze/master/input/forest1.jpg"),
    ("landscape", "mountain.png",
     "https://raw.githubusercontent.com/He-Zhang/image_dehaze/master/input/mountain.png"),
    ("landscape", "tree2.png",
     "https://raw.githubusercontent.com/JiamingMai/Color-Attenuation-Prior-Dehazing/master/inputImgs/tree2.png"),
    ("cityscape", "tiananmen1.png",
     "https://raw.githubusercontent.com/He-Zhang/image_dehaze/master/input/tiananmen1.png"),
    ("cityscape", "img_8766.jpg",
     "https://raw.githubusercontent.com/He-Zhang/image_dehaze/master/input/IMG_8766.jpg"),
]

CLEAN_DIRECT = [
    ("landscape", "kodim24.png", "http://r0k.us/graphics/kodak/kodak/kodim24.png"),
    ("scene", "kodim08.png", "http://r0k.us/graphics/kodak/kodak/kodim08.png"),
    ("bright", "kodim21_lighthouse.png", "http://r0k.us/graphics/kodak/kodak/kodim21.png"),
]

# ---- Wikimedia Commons file titles: (group, category, filename, title) -----
WIKIMEDIA = [
    ("real", "cityscape", "nyc_foggy_skyline.jpg", "File:NYC foggy skyline.jpg"),
    ("clean", "bright", "everest_snow.jpg",
     "File:Mount Everest as seen from Drukair2 PLW edit.jpg"),
    ("clean", "indoor", "banquet_room.jpg",
     "File:Banquet Room interior, Brother Hotel 20130212.jpg"),
]


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _resolve_wikimedia(title: str, width: int = MAX_DIM) -> str:
    """Resolve a Commons File: title to a downloadable (thumb) URL."""
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "imageinfo",
        "iiprop": "url", "iiurlwidth": str(width), "titles": title,
    })
    import json
    data = json.loads(_http_get("https://commons.wikimedia.org/w/api.php?" + q))
    page = list(data["query"]["pages"].values())[0]
    info = page.get("imageinfo")
    if not info:
        raise RuntimeError(f"no imageinfo for {title!r}")
    return info[0].get("thumburl") or info[0]["url"]


def _save_capped(raw: bytes, out_path: str) -> str:
    """Decode bytes, downscale so the long side <= MAX_DIM, write PNG/JPG."""
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError("could not decode image bytes")
    h, w = arr.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1.0:
        arr = cv2.resize(arr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, arr)
    return f"{arr.shape[1]}x{arr.shape[0]}"


def main() -> None:
    ok, fail = [], []

    def fetch(url: str, out_rel: str):
        out_path = os.path.join(ROOT, out_rel)
        if os.path.exists(out_path):
            ok.append(f"{out_rel}  (cached)")
            return
        try:
            dims = _save_capped(_http_get(url), out_path)
            ok.append(f"{out_rel}  ({dims})")
        except Exception as exc:  # noqa: BLE001 - report and continue
            fail.append(f"{out_rel}  <- {url}\n      {type(exc).__name__}: {exc}")

    for cat, name, url in REAL_DIRECT:
        fetch(url, f"data/real/{cat}/{name}")
    for cat, name, url in CLEAN_DIRECT:
        fetch(url, f"data/clean/{cat}/{name}")
    for group, cat, name, title in WIKIMEDIA:
        sub = "real" if group == "real" else "clean"
        out_rel = f"data/{sub}/{cat}/{name}"
        if os.path.exists(os.path.join(ROOT, out_rel)):
            ok.append(f"{out_rel}  (cached)")
            continue
        try:
            url = _resolve_wikimedia(title)
        except Exception as exc:  # noqa: BLE001
            fail.append(f"data/{sub}/{cat}/{name}  <- {title}\n      {type(exc).__name__}: {exc}")
            continue
        fetch(url, f"data/{sub}/{cat}/{name}")

    print("\n=== Downloaded ({}) ===".format(len(ok)))
    for line in ok:
        print("  OK   ", line)
    if fail:
        print("\n=== Failed ({}) ===".format(len(fail)))
        for line in fail:
            print("  FAIL ", line)
    print()
    if not ok:
        sys.exit("No images downloaded - check network access.")


if __name__ == "__main__":
    main()
