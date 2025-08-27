#!/bin/bash

# Dataframe2Visualization Setup Script
echo "🚀 Setting up Dataframe2Visualization..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python version $python_version is too old. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

echo "✅ Virtual environment created"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated"

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed successfully"

# Run tests to verify installation
echo "🧪 Running tests to verify installation..."
python -m pytest tests/ -v --tb=short

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "To run the application:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run the app: streamlit run app.py"
    echo "3. Deactivate when done: deactivate"
    echo ""
    echo "To run tests:"
    echo "source venv/bin/activate && python -m pytest tests/"
else
    echo "⚠️ Some tests failed, but setup completed. Check the output above for details."
fi
