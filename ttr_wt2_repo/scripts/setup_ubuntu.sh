#!/usr/bin/env bash
set -euo pipefail

# Basic bootstrap for a vanilla Ubuntu box.
# Assumes NVIDIA drivers + CUDA are already installed (e.g. AWS DLAMI) or installed separately.

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git \
  python3 \
  python3-venv \
  python3-pip \
  build-essential

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

# Install PyTorch CUDA wheel (example for CUDA 12.1).
# Adjust for your stack if needed.
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install project deps
python -m pip install -r requirements.txt
python -m pip install -e .

echo "✅ Setup complete. Activate with: source .venv/bin/activate"
