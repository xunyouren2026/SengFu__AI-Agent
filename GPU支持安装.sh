#!/bin/bash
# UFO AGI - GPU支持安装 (Linux/macOS)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "============================================"
echo "  UFO AGI - GPU支持安装 (CUDA)"
echo "============================================"
echo ""
echo "[提示] 需要NVIDIA显卡，约5GB下载"
echo ""

source "$PROJECT_DIR/.venv/bin/activate"

echo "[安装] PyTorch + CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if [ $? -ne 0 ]; then
    echo "[警告] CUDA安装失败，尝试CPU版本..."
    pip install torch torchvision torchaudio
fi

echo ""
echo "[完成] GPU支持安装完成!"
read -p "按回车键继续..."
