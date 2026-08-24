@echo off
echo ==========================================
echo Brite Spark 2026 - Problem 6
echo The Overpayment Signal
echo ==========================================

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Running pipeline...
python main.py

call .venv\Scripts\deactivate.bat

echo.
echo ==========================================
echo Pipeline complete! Check the output/ folder.
echo ==========================================
pause