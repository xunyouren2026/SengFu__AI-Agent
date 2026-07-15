@echo off
setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
set "VENV_PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

echo.
echo ============================================
echo   UFO AGI - HTML旧版界面启动
echo   天衡/Pendulum AGI Framework
echo ============================================
echo.

REM 检查环境
if not exist "%VENV_PYTHON%" (
    echo [错误] 虚拟环境不存在，请先运行: 环境安装配置.bat
    pause
    exit /b 1
)

echo [启动] 后端服务 (端口 8000)...
echo.
echo   HTML界面: http://localhost:8000/web/
echo   API文档:  http://localhost:8000/docs
echo.
echo   按 Ctrl+C 停止服务
echo ----------------------------------------
echo.

cd /d "%PROJECT_DIR%"
"%VENV_PYTHON%" main.py

if !errorlevel! neq 0 (
    echo.
    echo [错误] 服务异常退出
    pause
)

endlocal
