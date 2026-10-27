
#!/bin/bash

# This script automates the setup of a GPU development environment on Ubuntu.
# It automatically detects the user who ran the script and their home directory.
# Usage: sudo ./setup.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Dynamic User & Path Detection ---
if [ -z "$SUDO_USER" ]; then
    echo "❌ Error: This script must be run with sudo."
    exit 1
fi

LOCALUSER="$SUDO_USER"
# getent passwd finds the home directory regardless of whether it's in /home or /localhome
LOCALBASE=$(getent passwd "$LOCALUSER" | cut -d: -f6)

echo "--- System Detection ---"
echo "Detected User: $LOCALUSER"
echo "Detected Home: $LOCALBASE"

# Define Paths
ANACONDA_INSTALL_PATH="$LOCALBASE/anaconda3"
CUDA_HOME="/usr/local/cuda"

# --- 2. Architecture Detection & URLs ---
ARCH=$(uname -m)
echo "Detected Architecture: $ARCH"

if [ "$ARCH" = "x86_64" ]; then
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/13.4.3/local_installers/cuda_13.4.3_linux.run" 
    CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.27.0.42_cuda13-archive.tar.xz"
    ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2026.07-1-Linux-x86_64.sh"
elif [ "$ARCH" = "aarch64" ]; then
    CUDA_URL="https://developer.download.nvidia.com/compute/cuda/13.4.3/local_installers/cuda_13.4.3_linux_sbsa.run"
    CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-sbsa/cudnn-linux-sbsa-9.27.0.42_cuda13-archive.tar.xz"
    ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2026.07-1-Linux-aarch64.sh"
else
    echo "❌ Unsupported architecture: $ARCH"
    exit 1
fi

# --- 3. System Update and Prerequisites ---
# The DGX "baseos" repo (urm.nvidia.com/.../latest/baseos) frequently 404s on
# freshly-imaged DGX nodes, which makes EVERY `apt update` exit non-zero and
# aborts this script under `set -e` (e.g. at the Docker step).  Disable it when
# its Release file is unreachable; the working Ubuntu / CUDA / Docker repos
# cover everything this script installs.
DGX_SRC="/etc/apt/sources.list.d/dgx.sources"
if [ -f "$DGX_SRC" ]; then
    DGX_URI=$(awk -F': *' '/^URIs:/{print $2; exit}' "$DGX_SRC")
    if [ -n "$DGX_URI" ] && ! curl -fsI --max-time 10 "${DGX_URI%/}/dists/noble/Release" >/dev/null 2>&1; then
        echo "⚠️  DGX repo unreachable ($DGX_URI) -- disabling $DGX_SRC"
        mv "$DGX_SRC" "$DGX_SRC.disabled"
    fi
fi

echo "--- Updating package lists and installing prerequisites ---"
apt install -y --fix-broken
apt update || true
apt install -y wget curl fish git build-essential linux-headers-$(uname -r)
apt install -y infiniband-diags ibutils libibverbs1 librdmacm1
apt install -y nvitop

# --- 4. Install NVIDIA Driver & CUDA Toolkit ---
echo "--- Installing CUDA Toolkit ---"
wget -q --show-progress -O cuda_linux.run "$CUDA_URL"
# --silent: no UI; --samples: CUDA samples; --toolkit: CUDA toolkit.
# The driver flag is not passed.
sh cuda_linux.run --silent --samples --toolkit #--driver
rm cuda_linux.run

# --- 5. Install cuDNN ---
echo "--- Installing cuDNN ---"
wget -q --show-progress -O cudnn-archive.tar.xz "$CUDNN_URL"
tar -xf cudnn-archive.tar.xz
CUDNN_DIR=$(find . -maxdepth 1 -type d -name 'cudnn*-archive')
cp -P "$CUDNN_DIR/include/"* "$CUDA_HOME/include/"
cp -P "$CUDNN_DIR/lib/"* "$CUDA_HOME/lib64/"
rm -rf "$CUDNN_DIR" cudnn-archive.tar.xz

# --- 6. Install Anaconda ---
echo "--- Installing Anaconda to $ANACONDA_INSTALL_PATH ---"
wget -q --show-progress -O anaconda3.sh "$ANACONDA_URL"
# Run the installer. -b is batch mode, -u updates an existing install (so a
# re-run after a partial setup doesn't abort), -p is the path.
bash anaconda3.sh -b -u -p "$ANACONDA_INSTALL_PATH"
# Fix ownership so the user can use conda without sudo
chown -R "$LOCALUSER:$LOCALUSER" "$ANACONDA_INSTALL_PATH"
rm anaconda3.sh

# # --- 7. Install Docker Engine ---
# echo "--- Installing Docker Engine ---"
# install -m 0755 -d /etc/apt/keyrings
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
# chmod a+r /etc/apt/keyrings/docker.asc

# echo \
#   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
#   $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
#   tee /etc/apt/sources.list.d/docker.list > /dev/null
# apt update
# apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# # --- 8. Install NVIDIA Container Toolkit ---
# echo "--- Installing NVIDIA Container Toolkit ---"
# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
# curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
#   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#   tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
# apt update
# apt install -y nvidia-container-toolkit

# # --- 9. Final System Configuration ---
# echo "--- Finalizing Configuration ---"
# nvidia-ctk runtime configure --runtime=docker
# usermod -aG docker "$LOCALUSER"
# systemctl restart docker

# System-wide CUDA paths
tee /etc/profile.d/cuda.sh > /dev/null <<EOF
export PATH="$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
EOF

# Initialize Conda for the user's shells
sudo -u "$LOCALUSER" "$ANACONDA_INSTALL_PATH/bin/conda" init bash
if command -v fish >/dev/null 2>&1; then
    sudo -u "$LOCALUSER" "$ANACONDA_INSTALL_PATH/bin/conda" init fish
fi

echo "---------------------------------------------------------------------"
echo "✅ Setup complete for user: $LOCALUSER"
echo "✅ Anaconda installed at: $ANACONDA_INSTALL_PATH"
echo "---------------------------------------------------------------------"
echo "IMPORTANT: Please LOG OUT and LOG BACK IN to apply group and PATH changes."
echo "---------------------------------------------------------------------"
