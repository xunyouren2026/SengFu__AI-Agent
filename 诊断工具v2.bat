@echo off
chcp 65001 >nul
title AGI框架诊断工具 v2.0

echo.
echo  ================================================
echo     AGI 统一框架 - 前端数据问题诊断工具 v2.0
echo  ================================================
echo.

echo [1/7] 检查后端服务状态...
curl -s --connect-timeout 5 http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo     状态: ❌ 后端服务未运行
    echo     解决方案: 请先启动后端服务
    echo     命令: start-backend.bat
    echo.
    set BACKEND_RUNNING=0
) else (
    echo     状态: ✅ 后端服务运行中
    echo.
    set BACKEND_RUNNING=1
)

echo [2/7] 测试核心API端点...
if %BACKEND_RUNNING%==1 (
    curl -s --connect-timeout 5 http://localhost:8000/api/v1/system/metrics >nul 2>&1
    if %errorlevel% neq 0 (
        echo     /api/v1/system/metrics: ❌ 404/不可用
    ) else (
        echo     /api/v1/system/metrics: ✅ 可用
    )
    
    curl -s --connect-timeout 5 http://localhost:8000/api/v1/dashboard/stats >nul 2>&1
    if %errorlevel% neq 0 (
        echo     /api/v1/dashboard/stats: ❌ 404/不可用
    ) else (
        echo     /api/v1/dashboard/stats: ✅ 可用
    )
    
    curl -s --connect-timeout 5 http://localhost:8000/api/v1/training/jobs >nul 2>&1
    if %errorlevel% neq 0 (
        echo     /api/v1/training/jobs: ❌ 404/不可用
    ) else (
        echo     /api/v1/training/jobs: ✅ 可用
    )
) else (
    echo     跳过 - 后端未运行
)
echo.

echo [3/7] 检查API文档...
if %BACKEND_RUNNING%==1 (
    curl -s --connect-timeout 5 http://localhost:8000/openapi.json >nul 2>&1
    if %errorlevel% neq 0 (
        echo     OpenAPI文档: ❌ 不可访问
    ) else (
        echo     OpenAPI文档: ✅ 可访问
        echo     查看完整API: http://localhost:8000/docs
    )
) else (
    echo     跳过 - 后端未运行
)
echo.

echo [4/7] 检查前端页面...
if exist "web\pages\dashboard.html" (
    echo     dashboard.html: ✅ 存在
) else (
    echo     dashboard.html: ❌ 不存在
)

if exist "web\static\js\api-client-v2.js" (
    echo     api-client-v2.js: ✅ 存在 (新版本统一客户端)
) else (
    echo     api-client-v2.js: ❌ 不存在 - 请复制到 web\static\js\
)
echo.

echo [5/7] 检查Mock数据清理状态...
findstr /C:"displayMockData" web\pages\dashboard.html >nul 2>&1
if %errorlevel% neq 0 (
    echo     dashboard.html: ✅ 已清理Mock数据
) else (
    echo     dashboard.html: ⚠️ 仍有Mock数据引用
)

findstr /C:"Math.random()" web\pages\dashboard.html | findstr /C:"模拟" >nul 2>&1
if %errorlevel% neq 0 (
    echo     dashboard.html: ✅ 已移除随机数模拟
) else (
    echo     dashboard.html: ⚠️ 仍有Math.random()模拟
)
echo.

echo [6/7] 空状态提示检查...
findstr /C:"empty-state" web\pages\dashboard.html >nul 2>&1
if %errorlevel% equ 0 (
    echo     空状态提示: ✅ 已添加
) else (
    echo     空状态提示: ❌ 未添加
)
echo.

echo [7/7] 常见问题快速修复...
echo.
echo     ================================================
echo     问题诊断结果:
echo     ================================================
echo.

if %BACKEND_RUNNING%==0 (
    echo     ⚠️ 后端服务未运行！
    echo.
    echo     请按以下步骤启动:
    echo     1. 打开一个新的命令窗口
    echo     2. 进入项目目录
    echo     3. 运行: start-backend.bat
    echo.
) else (
    echo     ✅ 后端服务运行正常
    echo.
    echo     如果页面仍显示"暂无数据":
    echo     1. 检查浏览器控制台 (F12) 是否有错误
    echo     2. 检查Network标签页查看API请求状态
    echo     3. 确认API返回的数据格式是否正确
    echo.
)

echo     ================================================
echo     修复工具使用说明:
echo     ================================================
echo     1. 一键清理Mock数据: 运行前端修复.bat
echo     2. 启动后端服务: start-backend.bat  
echo     3. 启动前端服务: start-frontend.bat
echo     4. 查看API文档: http://localhost:8000/docs
echo     ================================================
echo.

echo 诊断完成！按任意键退出...
pause >nul
