#!/usr/bin/env bash
# Fetch the DehazeFormer demo model (self-contained model code + mixed-dataset
# weights) from the authors' HuggingFace Space. Idempotent: skips files already
# present. This is the modern/SOTA baseline used by src/run_sota.py.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$ROOT/external/DehazeFormer_Demo"
BASE="https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo"

mkdir -p "$DST/models" "$DST/saved_models"
[ -f "$DST/app.py" ]                      || curl -fsSL -o "$DST/app.py"                      "$BASE/raw/main/app.py"
[ -f "$DST/models/__init__.py" ]          || curl -fsSL -o "$DST/models/__init__.py"          "$BASE/raw/main/models/__init__.py"
[ -f "$DST/models/dehazeformer.py" ]      || curl -fsSL -o "$DST/models/dehazeformer.py"      "$BASE/raw/main/models/dehazeformer.py"
[ -f "$DST/saved_models/dehazeformer.pth" ] || curl -fsSL -o "$DST/saved_models/dehazeformer.pth" "$BASE/resolve/main/saved_models/dehazeformer.pth"

uv run --extra dehazeformer python - "$DST/saved_models/dehazeformer.pth" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
assert "state_dict" in sd and len(sd["state_dict"]) > 100, "weights look corrupt"
print("DehazeFormer demo model ready:", sys.argv[1])
PY
