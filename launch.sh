#!/bin/bash

echo "🏥 AI-Based No-Show Appointment Prediction Tool"
echo "================================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if requirements are installed
echo "🔍 Checking dependencies..."
python3 -c "import streamlit, pandas, numpy, sklearn, plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    python3 -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies are ready"
echo ""
echo "🚀 Starting the application..."
echo "📱 The app will open in your default web browser"
echo "🔄 To stop the app, press Ctrl+C in this terminal"
echo ""
echo "================================================"

# Run the Streamlit app
python3 -m streamlit run app.py --server.port 8501

