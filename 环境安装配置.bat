@echo off
setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

echo.
echo ============================================
echo   UFO AGI - 环境安装配置
echo   天衡/Pendulum AGI Framework
echo ============================================
echo.

REM 检查Python
echo [1/5] 检测Python环境...
where python >nul 2>&1
if !errorlevel! neq 0 (
    echo [错误] 未找到Python，请先安装 Python 3.9+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo [成功] Python已就绪
echo.

REM 创建虚拟环境
echo [2/5] 创建虚拟环境...
if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo [成功] 虚拟环境已创建
) else (
    echo [成功] 虚拟环境已存在
)
echo.

REM 安装依赖
echo [3/5] 安装基础依赖...
echo (首次安装可能需要5-10分钟)
"%VENV_PYTHON%" -m pip install --upgrade pip -q
"%VENV_PIP%" install -r "%PROJECT_DIR%requirements.txt"
if !errorlevel! neq 0 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)
echo [成功] 依赖安装完成
echo.

REM 初始化数据库
echo [4/5] 初始化数据库...
cd /d "%PROJECT_DIR%"
"%VENV_PYTHON%" main.py --init-db
if !errorlevel! neq 0 (
    echo [错误] 数据库初始化失败
    pause
    exit /b 1
)
echo [成功] 数据库初始化完成
echo.

REM 设置环境变量
echo [5/5] 设置环境变量...
setx UFO_PROJECT_DIR "%PROJECT_DIR%" >nul 2>&1
setx UFO_VENV_PYTHON "%VENV_PYTHON%" >nul 2>&1
setx UFO_BACKEND_PORT "8000" >nul 2>&1
setx UFO_FRONTEND_PORT "3000" >nul 2>&1
echo [成功] 环境变量已设置
echo.

echo ============================================
echo   环境配置完成!
echo ============================================
echo.
echo   虚拟环境: %VENV_DIR%
echo   Python:   %VENV_PYTHON%
echo.
echo   现在可以运行:
echo     HTML启动.bat  - 启动HTML旧版界面
echo     TSX启动.bat   - 启动TSX新版界面
echo.
pause
endlocal
