@echo off
REM ===================================================================
REM  SHM Motion Tracker - First-Time Setup (Windows)
REM  University of Phayao - Physics Department
REM ===================================================================
REM  Run this once on a new computer to install everything
REM  Requires Python 3.12 (recommended) installed first
REM ===================================================================

title Setup - SHM Motion Tracker
color 0B

echo.
echo ===============================================
echo   SETUP - SHM MOTION TRACKER
echo   University of Phayao - Physics Department
echo ===============================================
echo.

cd /d "%~dp0"

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python not found
    echo.
    echo Please install Python 3.12 first:
    echo https://www.python.org/downloads/release/python-3128/
    echo.
    echo IMPORTANT: Check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version') do set PYVER=%%i
echo        Python %PYVER% found
echo.

echo %PYVER% | findstr /r "^3\.1[34]" >nul
if not errorlevel 1 (
    echo [WARNING] Python %PYVER% may be too new for scipy
    echo            Recommended: Python 3.12
    echo.
    set /p CONTINUE="Continue anyway? (Y/N): "
    if /i not "%CONTINUE%"=="Y" exit /b 1
    echo.
)

echo [2/4] Creating virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo        venv exists - skipping
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
    echo        venv created
)
echo.

echo [3/4] Activating venv...
call venv\Scripts\activate.bat
echo.

echo [4/4] Installing libraries...
echo        First time install takes 3-5 minutes...
echo.
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Library installation failed
    echo         Please check internet connection
    echo         Or try Python 3.12 instead
    pause
    exit /b 1
)

echo.
echo ===============================================
echo   SETUP COMPLETE!
echo ===============================================
echo.
echo Now you can use the program by
echo double-clicking "start_app.bat"
echo.
pause
