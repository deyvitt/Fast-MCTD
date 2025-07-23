#!/bin/bash

# Fast-MCTD with MoR Installation Script
# Compatible with Linux and macOS

set -e

echo "🚀 Installing Fast-MCTD with Mixture of Recursions..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "   CUDA Version: $cuda_version"
    USE_CUDA=true
else
    echo "⚠️  No NVIDIA GPU detected, installing CPU-only version"
    USE_CUDA=false
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv fast_mctd_env
source fast_mctd_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch
echo "🔥 Installing PyTorch..."
if [ "$USE_CUDA" = true ]; then
    pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install core requirements
echo "📋 Installing core requirements..."
pip install -r requirements.txt

# Install optional GPU packages
if [ "$USE_CUDA" = true ]; then
    echo "🚀 Installing GPU-specific packages..."
    pip install faiss-gpu>=1.7.4
    pip install deepspeed>=0.9.0
fi

# Install package in development mode
echo "🔧 Installing Fast-MCTD package..."
pip install -e .

# Verify installation
echo "🧪 Verifying installation..."
python -c "
import torch
import torchvision
import numpy as np
import scipy
import matplotlib
import tqdm
import wandb
import lpips
print('✅ All core packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current CUDA device: {torch.cuda.get_device_name()}')
"

echo "🎉 Installation completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "source fast_mctd_env/bin/activate"
echo ""
echo "To test the installation, run:"
echo "python -c 'from fast_mctd import create_mor_fast_mctd_system; print(\"Fast-MCTD with MoR ready!\")'"
