@echo off
setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"

echo.
echo ============================================
echo   UFO AGI - 前端修复
echo   天衡/Pendulum AGI Framework
echo ============================================
echo.

cd /d "%PROJECT_DIR%\frontend"

echo [1/3] 删除 node_modules...
if exist "node_modules" (
    rmdir /s /q node_modules
    echo [成功] 已删除
)

echo [2/3] 删除 package-lock.json...
if exist "package-lock.json" (
    del /f package-lock.json
    echo [成功] 已删除
)

echo [3/3] 重新安装依赖...
npm cache clean --force
npm install
if !errorlevel! neq 0 (
    echo [错误] 安装失败
    pause
    exit /b 1
)

echo.
echo [完成] 前端修复完成!
pause
endlocal
