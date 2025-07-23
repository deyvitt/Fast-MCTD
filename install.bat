
@echo off
echo 🚀 Installing Fast-MCTD with Mixture of Recursions...

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python version: %python_version%

REM Check for CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo ⚠️  No NVIDIA GPU detected, installing CPU-only version
    set USE_CUDA=false
) else (
    echo ✅ NVIDIA GPU detected
    set USE_CUDA=true
)

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv fast_mctd_env
call fast_mctd_env\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install PyTorch
echo 🔥 Installing PyTorch...
if "%USE_CUDA%"=="true" (
    pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
) else (
    pip install torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
)

REM Install core requirements
echo 📋 Installing core requirements...
pip install -r requirements.txt

REM Install optional GPU packages
if "%USE_CUDA%"=="true" (
    echo 🚀 Installing GPU-specific packages...
    pip install faiss-gpu>=1.7.4
    pip install deepspeed>=0.9.0
)

REM Install package in development mode
echo 🔧 Installing Fast-MCTD package...
pip install -e .

REM Verify installation
echo 🧪 Verifying installation...
python -c "import torch; import numpy as np; import matplotlib; print('✅ Installation verified!')"

echo 🎉 Installation completed successfully!
echo.
echo To activate the environment, run:
echo fast_mctd_env\Scripts\activate.bat
echo.
pause
