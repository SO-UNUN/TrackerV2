@echo off
REM ===================================================================
REM  SHM Motion Tracker - Quick Start (Windows)
REM  University of Phayao - Physics Department
REM ===================================================================

title SHM Motion Tracker
color 0E

echo.
echo ===============================================
echo   SHM MOTION TRACKER
echo   University of Phayao - Physics Department
echo ===============================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found
    echo.
    echo Please run setup.bat first to install
    echo.
    pause
    exit /b 1
)

echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat

python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Streamlit not installed
    echo Please run setup.bat first
    echo.
    pause
    exit /b 1
)

echo [2/2] Launching app...
echo.
echo Browser will open at: http://localhost:8501
echo.
echo ===============================================
echo   To close: close this window or press Ctrl+C
echo ===============================================
echo.

streamlit run app.py

pause
