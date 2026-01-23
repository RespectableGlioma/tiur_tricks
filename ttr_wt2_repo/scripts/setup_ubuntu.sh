#!/usr/bin/env bash
set -euo pipefail

# Basic bootstrap for a vanilla Ubuntu box (e.g. AWS P4d).
# Installs NVIDIA drivers, Fabric Manager, Python, and Project deps.

echo ">>> Setting up NVIDIA Drivers & Fabric Manager..."

# 1. Detect OS for repos
. /etc/os-release
DISTRO=$ID$VERSION_ID
DISTRO_NODOT=${DISTRO//./}

# 2. Add NVIDIA repo
wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO_NODOT/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

sudo apt-get update

# 3. Install Drivers + CUDA + Fabric Manager
# (Using 550 drivers + CUDA 12.4 for Ubuntu 24.04 compatibility)
sudo apt-get install -y --no-install-recommends \
    cuda-drivers-550 \
    cuda-toolkit-12-4 \
    nvidia-fabricmanager-550 \
    nvidia-utils-550 \
    git \
    python3 \
    python3-venv \
    python3-pip \
    build-essential

# 4. Enable Fabric Manager (CRITICAL for A100s)
sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager

echo ">>> Setting up Python Environment..."

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel

# Install PyTorch CUDA wheel (CUDA 12.4)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install project deps
python -m pip install -r requirements.txt
python -m pip install -e .

echo "✅ Setup complete. Activate with: source .venv/bin/activate"
