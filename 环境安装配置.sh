#!/bin/bash
# UFO AGI - 环境安装配置 (Linux/macOS)

echo ""
echo "============================================"
echo "  UFO AGI - 环境安装配置"
echo "  天衡/Pendulum AGI Framework"
echo "============================================"
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

echo "[1/5] 检测Python..."
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3"
    exit 1
fi
python3 --version
echo ""

echo "[2/5] 创建虚拟环境..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
echo "[成功] 虚拟环境就绪"
echo ""

echo "[3/5] 安装基础依赖..."
source "$VENV_DIR/bin/activate"
pip install -r "$PROJECT_DIR/requirements.txt"
echo "[成功] 依赖安装完成"
echo ""

echo "[4/5] 初始化数据库..."
cd "$PROJECT_DIR"
python main.py --init-db
echo "[成功] 数据库初始化完成"
echo ""

echo "[5/5] 设置环境变量..."
echo "export UFO_PROJECT_DIR=$PROJECT_DIR" >> ~/.bashrc
echo "export UFO_VENV_PYTHON=$VENV_PYTHON" >> ~/.bashrc
echo "[成功] 环境变量已添加到 ~/.bashrc"
echo ""

echo "============================================"
echo "  环境配置完成!"
echo "============================================"
echo ""
echo "  运行: source ~/.bashrc"
echo "  然后: ./HTML启动.sh 或 ./TSX启动.sh"
echo ""
read -p "按回车键继续..."
