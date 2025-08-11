@echo off
echo 🧠 Starting REAL AI Smart Parking Server...
echo.

REM Deactivate any virtual environment
set VIRTUAL_ENV=
set PATH=%PATH:;%VIRTUAL_ENV%\Scripts=%

REM Change to project directory
cd /d "C:\Users\admin\Dropbox\PC\Desktop\thuctaptotnghiep"

REM Run with system Python
python real_ai_direct.py

pause
