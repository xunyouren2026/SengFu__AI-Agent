@echo off
chcp 65001 >nul
REM =============================================================================
REM UFO AGI Unified Framework - Windows Startup Script
REM =============================================================================

setlocal EnableDelayedExpansion

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "LOG_DIR=%PROJECT_DIR%logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [INFO] UFO AGI Framework Startup Script
echo [INFO] Project: %PROJECT_DIR%

if "%~1"=="--setup" goto do_setup
if "%~1"=="--stop" goto do_stop
if "%~1"=="--help" goto do_help
goto do_start

:do_setup
echo.
echo ============================================
echo   UFO AGI Framework Setup
echo ============================================
echo.

REM Check Python
where python >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)
echo [OK] Python found

REM Create virtual environment
if not exist "%VENV_DIR%" (
    echo [INFO] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Check virtual environment Python
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment Python not found: %VENV_PYTHON%
    pause
    exit /b 1
)
echo [OK] Virtual environment ready

REM Install dependencies
echo.
echo [INFO] Installing dependencies to virtual environment...
echo          This may take 5-10 minutes...
echo.

"%VENV_PYTHON%" -m pip install --upgrade pip setuptools wheel

if !errorlevel! neq 0 (
    echo [WARNING] pip upgrade failed, continuing...
)

"%VENV_PIP%" install -r "%PROJECT_DIR%requirements.txt"

if !errorlevel! neq 0 (
    echo [ERROR] Failed to install dependencies
    echo [INFO] Trying again with verbose output...
    "%VENV_PIP%" install -r "%PROJECT_DIR%requirements.txt" --verbose
    pause
    exit /b 1
)

echo [OK] Dependencies installed

REM Create config
if not exist "%PROJECT_DIR%.env" (
    if exist "%PROJECT_DIR%.env.example" (
        copy "%PROJECT_DIR%.env.example" "%PROJECT_DIR%.env" >nul
        echo [OK] Config file created: .env
    )
)

REM Init database
echo.
echo [INFO] Initializing database...
cd /d "%PROJECT_DIR%"
"%VENV_PYTHON%" main.py --init-db

if !errorlevel! neq 0 (
    echo [ERROR] Database initialization failed
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo Run 'start.bat' to start the application
echo.
pause
goto end

:do_start
REM Check if virtual environment exists
if not exist "%VENV_DIR%" (
    echo [INFO] First run detected, starting setup...
    goto do_setup
)

REM Check if virtual environment Python exists
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment corrupted. Please run: start.bat --setup
    pause
    exit /b 1
)

echo.
echo ============================================
echo   UFO AGI Framework
echo ============================================
echo.
echo   Web UI:   http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
echo   Press Ctrl+C to stop
echo.

cd /d "%PROJECT_DIR%"
"%VENV_PYTHON%" main.py

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Application exited with error code !errorlevel!
    pause
)

goto end

:do_stop
echo [INFO] Stopping application...
taskkill /IM python.exe /F >nul 2>&1
echo [OK] Done
goto end

:do_help
echo UFO AGI Framework Startup Script
echo.
echo Usage: start.bat [option]
echo.
echo   start.bat         Start application
echo   start.bat --setup  Run setup (install dependencies)
echo   start.bat --stop   Stop application
echo   start.bat --help   Show this help
echo.
goto end

:end
endlocal
