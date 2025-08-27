@echo off
REM Dataframe2Visualization Setup Script for Windows
echo 🚀 Setting up Dataframe2Visualization...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% detected

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment created

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment activated

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed successfully

REM Run tests to verify installation
echo 🧪 Running tests to verify installation...
python -m pytest tests/ -v --tb=short

if errorlevel 1 (
    echo.
    echo ⚠️ Some tests failed, but setup completed. Check the output above for details.
) else (
    echo.
    echo 🎉 Setup completed successfully!
)

echo.
echo To run the application:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Run the app: streamlit run app.py
echo 3. Deactivate when done: deactivate
echo.
echo To run tests:
echo venv\Scripts\activate.bat ^&^& python -m pytest tests/
echo.
pause
