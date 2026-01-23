#!/usr/bin/env bash
set -euo pipefail

# Basic bootstrap for a vanilla Ubuntu box (e.g. AWS P4d).
# Installs NVIDIA drivers, Fabric Manager, Python, and Project deps.

echo ">>> Setting up NVIDIA Drivers & Fabric Manager..."

# 1. Detect OS for information only
. /etc/os-release
echo ">>> Setting up NVIDIA Drivers on $ID $VERSION_ID..."

# 2. Install Dependencies (Python + Build Tools)
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  git \
  python3 \
  python3-venv \
  python3-pip \
  build-essential \
  linux-headers-$(uname -r) \
  ubuntu-drivers-common

echo ">>> Attempting install via ubuntu-drivers (recommended alternative)..."
sudo ubuntu-drivers install --gpgpu

# Install CUDA Toolkit (Runfile) - safer than apt to avoid driver conflicts
echo ">>> Installing CUDA Toolkit 12.1..."
if [ ! -f "cuda_12.1.0_530.30.02_linux.run" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
fi
sudo sh cuda_12.1.0_530.30.02_linux.run --silent --toolkit

# Install Fabric Manager
# We still need the repo for this usually.
# Note: Using 2404 repo since we are on Noble
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Detect installed driver version to pick FM
echo ">>> Detecting driver version..."
# Try to load kernel module if not loaded (avoids reboot in some cases)
sudo modprobe nvidia || true

DRIVER_VER=""
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | cut -d. -f1 || true)
fi

if [ -z "$DRIVER_VER" ]; then
    echo ">>> nvidia-smi failed or not found. Attempting to detect driver from dpkg..."
    # Parse installed package name, e.g. nvidia-driver-535
    DRIVER_VER=$(dpkg-query -W -f='${Package}\n' 'nvidia-driver-*' 2>/dev/null | grep -oE '[0-9]+' | head -n 1 || true)
fi

if [ -z "$DRIVER_VER" ]; then
    echo "WARNING: Could not detect NVIDIA driver version. Fabric Manager install might fail."
    echo "Please REBOOT and run this script again."
    exit 1
fi

echo ">>> Detected installed driver major version: $DRIVER_VER"
sudo apt-get install -y nvidia-fabricmanager-${DRIVER_VER}
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
