@echo off
setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
set "VENV_PIP=%PROJECT_DIR%.venv\Scripts\pip.exe"

echo.
echo ============================================
echo   UFO AGI - GPU支持安装 (CUDA)
echo   天衡/Pendulum AGI Framework
echo ============================================
echo.
echo [提示] 需要NVIDIA显卡，约5GB下载
echo.

echo [安装] PyTorch + CUDA 11.8...
"%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if !errorlevel! neq 0 (
    echo [警告] CUDA安装失败，尝试CPU版本...
    "%VENV_PIP%" install torch torchvision torchaudio
)

echo.
echo [完成] GPU支持安装完成!
pause
endlocal
