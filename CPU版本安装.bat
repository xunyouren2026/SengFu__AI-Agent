@echo off
setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
set "VENV_PIP=%PROJECT_DIR%.venv\Scripts\pip.exe"

echo.
echo ============================================
echo   UFO AGI - CPU版本安装
echo   天衡/Pendulum AGI Framework
echo ============================================
echo.
echo [提示] 无需显卡，约2GB下载
echo.

echo [安装] PyTorch CPU版本...
"%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo [完成] CPU版本安装完成!
pause
endlocal
