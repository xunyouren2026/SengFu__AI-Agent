#!/bin/bash
# UFO AGI - CPU版本安装 (Linux/macOS)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "============================================"
echo "  UFO AGI - CPU版本安装"
echo "============================================"
echo ""
echo "[提示] 无需显卡，约2GB下载"
echo ""

source "$PROJECT_DIR/.venv/bin/activate"

echo "[安装] PyTorch CPU版本..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "[完成] CPU版本安装完成!"
read -p "按回车键继续..."
