#!/bin/bash
# UFO AGI - TSX新版界面启动 (Linux/macOS)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"

echo ""
echo "============================================"
echo "  UFO AGI - TSX新版界面启动"
echo "============================================"
echo ""

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[错误] 虚拟环境不存在，请先运行: ./环境安装配置.sh"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "[错误] 未找到Node.js，请安装 Node.js 20+"
    exit 1
fi

echo "[1/2] 启动后端API (端口 8000)..."
cd "$PROJECT_DIR"
source "$PROJECT_DIR/.venv/bin/activate"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
sleep 3
echo "[成功] 后端已启动"
echo ""

echo "[2/2] 启动TSX前端 (端口 3000)..."
cd "$PROJECT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    echo "[提示] 首次运行，安装前端依赖..."
    npm install
fi
npm run dev &
FRONTEND_PID=$!
sleep 3
echo "[成功] 前端已启动"
echo ""

echo "============================================"
echo "  TSX启动完成!"
echo "============================================"
echo ""
echo "  打开浏览器: http://localhost:3000"
echo ""
read -p "按回车键打开浏览器..."
xdg-open http://localhost:3000 2>/dev/null || open http://localhost:3000 2>/dev/null

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
