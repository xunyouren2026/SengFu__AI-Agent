#!/bin/bash
# UFO AGI - HTML旧版界面启动 (Linux/macOS)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

echo ""
echo "============================================"
echo "  UFO AGI - HTML旧版界面启动"
echo "============================================"
echo ""

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[错误] 虚拟环境不存在，请先运行: ./环境安装配置.sh"
    exit 1
fi

echo "[启动] 后端服务 (端口 8000)..."
echo ""
echo "  HTML界面: http://localhost:8000/web/"
echo "  API文档:  http://localhost:8000/docs"
echo ""
echo "  按 Ctrl+C 停止服务"
echo "----------------------------------------"
echo ""

cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
python main.py
