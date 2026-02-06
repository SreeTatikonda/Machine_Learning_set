#!/bin/bash

echo "=================================================="
echo "  📧 Email Classifier - Setup Script"
echo "=================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Create virtual environment (optional)
read -p "Do you want to create a virtual environment? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
else
    echo "⚠️  Skipping virtual environment creation"
fi

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --break-system-packages

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "Verifying installation..."
python3 src/test_classifier.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "  ✅ Setup Complete!"
    echo "=================================================="
    echo ""
    echo "Quick Start Options:"
    echo ""
    echo "1. Web Interface (Recommended):"
    echo "   python src/app.py"
    echo "   Then open: http://localhost:5000"
    echo ""
    echo "2. Command Line:"
    echo "   python src/classify_cli.py"
    echo ""
    echo "3. Run Tests:"
    echo "   python src/test_classifier.py"
    echo ""
else
    echo "❌ Setup verification failed"
    exit 1
fi
