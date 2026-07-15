#!/bin/bash
# AGI Unified Framework - Web Server Launcher (Linux/Mac)

echo "============================================"
echo "   AGI Unified Framework - Web Server"
echo "============================================"
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到Python3，请先安装Python 3.x"
    echo "Ubuntu/Debian: sudo apt install python3"
    echo "Mac: brew install python3"
    exit 1
fi

echo "[信息] 正在启动服务器..."
echo ""

# 启动服务器
python3 start_server.py "$@"
