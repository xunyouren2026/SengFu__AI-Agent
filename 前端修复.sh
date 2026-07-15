#!/bin/bash
# UFO AGI - 前端修复 (Linux/macOS)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "============================================"
echo "  UFO AGI - 前端修复"
echo "============================================"
echo ""

cd "$PROJECT_DIR/frontend"

echo "[1/3] 删除 node_modules..."
rm -rf node_modules
echo "[成功]"

echo "[2/3] 删除 package-lock.json..."
rm -f package-lock.json
echo "[成功]"

echo "[3/3] 重新安装依赖..."
npm cache clean --force
npm install
echo "[成功]"

echo ""
echo "[完成] 前端修复完成!"
read -p "按回车键继续..."
