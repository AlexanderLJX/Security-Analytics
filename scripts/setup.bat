@echo off
echo 🚀 Setting up Phishing Detection System...

REM Check Python version
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.9 or higher.
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Check for CUDA
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo CUDA detected, installing GPU dependencies...
    pip install ".[gpu]"
) else (
    echo No CUDA detected, skipping GPU dependencies
)

REM Copy environment template
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.template .env
    echo ⚠️  Please edit .env file with your API keys
)

REM Create necessary directories
echo Creating directories...
mkdir data 2>nul
mkdir checkpoints 2>nul
mkdir logs 2>nul
mkdir results 2>nul

REM Create sample dataset if it doesn't exist
if not exist "data\phishing_emails.csv" (
    echo Creating sample dataset...
    python main.py create-sample --output-path ./data/phishing_emails.csv --num-samples 1000
)

echo ✅ Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Run: python main.py train --help
echo 3. Start with: python main.py create-sample --output-path ./data/phishing_emails.csv

pause