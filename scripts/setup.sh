#!/bin/bash

# Setup script for phishing detection system
set -e

echo "🚀 Setting up Phishing Detection System..."

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install GPU dependencies if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing GPU dependencies..."
    pip install ".[gpu]"
else
    echo "No CUDA detected, skipping GPU dependencies"
fi

# Copy environment template
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env file with your API keys"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data checkpoints logs results

# Create sample dataset if it doesn't exist
if [ ! -f "data/phishing_emails.csv" ]; then
    echo "Creating sample dataset..."
    python main.py create-sample --output-path ./data/phishing_emails.csv --num-samples 1000
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: python main.py train --help"
echo "3. Start with: python main.py create-sample --output-path ./data/phishing_emails.csv"