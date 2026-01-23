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
  awscli \
  linux-headers-$(uname -r)

# 3. Install Drivers from AWS S3 (Recommended for P4d)
# This avoids repo issues and ensures AWS compatibility.
echo ">>> Downloading NVIDIA Driver from AWS S3..."
mkdir -p /tmp/nvidia-driver
aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ /tmp/nvidia-driver --no-sign-request

echo ">>> Installing NVIDIA Driver..."
# Find the .run file (should be only one)
DRIVER_RUN_FILE=$(find /tmp/nvidia-driver -name "NVIDIA-Linux-x86_64*.run" | head -n 1)

if [ -z "$DRIVER_RUN_FILE" ]; then
    echo "ERROR: Could not find NVIDIA driver .run file downloaded from S3."
    exit 1
fi

sudo chmod +x "$DRIVER_RUN_FILE"
# Silent install, accept license, no kernel module rebuild prompt, no X check
sudo "$DRIVER_RUN_FILE" -s --no-questions --accept-license --disable-nouveau

# 4. Install CUDA Toolkit (Runfile) to match driver
# We'll install CUDA 12.1 (compatible with PyTorch 2.1+)
# NOTE: Driver is already installed, so we install toolkit ONLY.
CUDA_VERSION="12.1.0"
CUDA_RUNFILE="cuda_${CUDA_VERSION}_530.30.02_linux.run"
if [ ! -f "$CUDA_RUNFILE" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/$CUDA_RUNFILE
fi
sudo sh $CUDA_RUNFILE --silent --toolkit

# 5. Install Fabric Manager (Critical for A100/P4d)
# We need to find the version matching the installed driver.
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
echo ">>> Detected Driver Version: $DRIVER_VERSION"
# We'll attempt to install via apt, assuming the repo has a matching version, 
# OR use the specific AWS recommendation if available. 
# Since we installed driver manually, apt might be tricky for FM. 
# BUT often AWS DLAMI uses the repo. Let's try to add the repo just for FM.

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
# Fabric manager version must match driver version exactly (major.minor.patch) or at least major.minor
# For manual driver installs, this is painful. 
# SIMPLER PATH: Just use 'ubuntu-drivers' if possible? 
# The user prompt suggested 'aws s3 cp' for *drivers*.
# Let's try to trust the user's snippet for the driver part.

# For Fabric Manager with manual driver install, we usually need the .deb matching the driver.
# If we used the AWS driver, we might be on a version not perfectly in the Ubuntu repo.
# However, for P4d instances, the "AWS-provided" driver is usually very standard.

# Let's fallback to the second user suggestion if the S3 method is complex: 
# "For Ubuntu, sudo ubuntu-drivers --gpgpu install is an alternative."
# This is much cleaner if it pulls the right headless driver.

# Actually, let's try the S3 method as requested first, but we need Fabric Manager.
# AWS usually bundles FM or expects you to pull it from their repo.

# RE-STRATEGY based on user input:
# 1. Use `ubuntu-drivers` (standard, easy) OR `aws s3` (fast, robust). 
# Let's go with `aws s3` for the driver as requested.
# 2. For CUDA, use the runfile (decoupled from driver).
# 3. For Fabric Manager, it's tricky with manual .run drivers.
#    Wait, on AWS P4d, the DLAMI is best. But assuming vanilla Ubuntu:
#    If we install driver 535 via .run, we need FM 535 via apt.

# Let's try to stick to the repo method but FIX the package names?
# The user said "E: Unable to locate package cuda-toolkit-12-4".
# That likely means the repo wasn't added correctly or `apt update` failed silently.
# But the AWS S3 advice is solid. Let's switch to that.

# ...Actually, if we use the .run file, we avoid the apt dependency hell.
# But Fabric Manager IS required for A100.
# "sudo apt-get install nvidia-fabricmanager-535" works if the repo is there.

# Let's refine the script to use the S3 driver + apt for FM/CUDA? No, mixing is bad.
# Let's use the `ubuntu-drivers` tool as the user mentioned as an *alternative*.
# `sudo ubuntu-drivers install --gpgpu` 
# This is usually the safest "vanilla ubuntu" way.

echo ">>> Attempting install via ubuntu-drivers (recommended alternative)..."
sudo apt-get install -y ubuntu-drivers-common
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
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
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
