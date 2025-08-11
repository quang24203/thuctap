@echo off
title Real AI Smart Parking Server

echo.
echo ========================================
echo 🧠 REAL AI SMART PARKING SERVER
echo ========================================
echo.

REM Clear any virtual environment
set VIRTUAL_ENV=
set CONDA_DEFAULT_ENV=
set PYTHONPATH=

REM Use system Python directly
echo 🔍 Checking Python...
python --version
if errorlevel 1 (
    echo ❌ Python not found in PATH
    pause
    exit /b 1
)

echo.
echo 🚀 Starting Real AI Server...
echo 🌐 Will open: http://localhost:8000
echo.

REM Change to project directory and run
cd /d "C:\Users\admin\Dropbox\PC\Desktop\thuctaptotnghiep"
python simple_real_ai.py

echo.
echo 🛑 Server stopped
pause
