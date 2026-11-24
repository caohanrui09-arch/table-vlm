#!/usr/bin/env bash
# Prepare a single-environment demo on AutoDL (or any CUDA Linux host).
#
#   bash scripts/setup_demo_autodl.sh
#
# Environment knobs:
#   PYTHON=python3        interpreter used to create .venv
#   SKIP_TORCH=1          keep an already-installed CUDA torch
#   TRANSPORT=modelscope  modelscope (default) or curl (Hugging Face)
#   SKIP_DOWNLOAD=1       do not download the 4.25GB base weight
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "python3 is required; activate the image's Python or install 3.10+." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "==> creating .venv"
  "$PYTHON" -m venv .venv
fi
VENV_PY="$ROOT/.venv/bin/python"
"$VENV_PY" -m pip install --upgrade pip

if [ "${SKIP_TORCH:-0}" != "1" ]; then
  echo "==> installing torch 2.6.0 + cu124 (about 2.5GB)"
  "$VENV_PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
fi

echo "==> installing demo, data and worker dependencies"
"$VENV_PY" -m pip install \
  -r requirements-data.txt -r requirements-demo.txt \
  transformers==4.57.1 peft==0.18.1 accelerate==1.11.0

if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  if [ -f models/Qwen3-VL-2B-Instruct/model.safetensors ]; then
    echo "==> base weight already present; verifying"
  else
    echo "==> downloading the pinned base weight (about 4.25GB)"
    "$VENV_PY" -m table_vlm.download_model \
      --output models/Qwen3-VL-2B-Instruct \
      --metadata-manifest research/qwen2b-fixed-manifest.json \
      --transport "${TRANSPORT:-modelscope}"
  fi
fi

echo "==> preflight"
"$VENV_PY" -m table_vlm.doctor --deep --require-cuda
echo
echo "Ready. Start the UI with: bash scripts/run_demo_autodl.sh"
