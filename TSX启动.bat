@echo off
setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

echo.
echo ============================================
echo   UFO AGI - TSX新版界面启动
echo   天衡/Pendulum AGI Framework
echo ============================================
echo.

REM 检查环境
if not exist "%VENV_PYTHON%" (
    echo [错误] 虚拟环境不存在，请先运行: 环境安装配置.bat
    pause
    exit /b 1
)

where node >nul 2>&1
if !errorlevel! neq 0 (
    echo [错误] 未找到Node.js，请安装 Node.js 20+
    echo 下载地址: https://nodejs.org/
    pause
    exit /b 1
)

REM 启动后端
echo [1/2] 启动后端API (端口 8000)...
cd /d "%PROJECT_DIR%"
start /min "" cmd /c ""%VENV_PYTHON%" -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
ping -n 4 127.0.0.1 >nul
echo [成功] 后端已启动
echo.

REM 启动前端
echo [2/2] 启动TSX前端 (端口 3000)...
cd /d "%PROJECT_DIR%\frontend"
if not exist "node_modules" (
    echo [提示] 首次运行，安装前端依赖...
    call npm install
)
start "" cmd /c "npm run dev"
ping -n 5 127.0.0.1 >nul
echo [成功] 前端已启动
echo.

echo ============================================
echo   TSX启动完成!
echo ============================================
echo.
echo   打开浏览器: http://localhost:3000
echo.
pause
start http://localhost:3000
endlocal
