#!/bin/bash
set -ex

# Script to install NVIDIA Drivers, CUDA, and Fabric Manager on AWS P4d instances (Ubuntu 20.04/22.04).
# A100s require Fabric Manager to function correctly on AWS.

echo "Detecting OS..."
. /etc/os-release
DISTRO=$ID$VERSION_ID
DISTRO_NODOT=${DISTRO//./}

echo "Setting up NVIDIA repositories for $DISTRO..."
wget https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO_NODOT/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
rm cuda-keyring_1.0-1_all.deb
sudo apt-get update

echo "Installing NVIDIA Drivers (535 branch) and CUDA 12.1..."
# Using 535 drivers as they are stable for A100.
sudo apt-get install -y cuda-drivers-535 cuda-toolkit-12-1

echo "Installing NVIDIA Fabric Manager..."
# Fabric Manager version must match driver version.
sudo apt-get install -y nvidia-fabricmanager-535

echo "Starting Fabric Manager..."
sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager

echo "Verifying installation..."
nvidia-smi

echo "DONE. If nvidia-smi failed or showed no devices, please REBOOT the instance: sudo reboot"
