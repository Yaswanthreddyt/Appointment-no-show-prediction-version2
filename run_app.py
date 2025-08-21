#!/usr/bin/env python3
"""
Launcher script for the AI-Based No-Show Appointment Prediction Tool
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'pandas', 'numpy', 'scikit-learn', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install them using:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def main():
    """Main launcher function"""
    print("🏥 AI-Based No-Show Appointment Prediction Tool")
    print("=" * 50)
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("❌ Error: app.py not found in current directory")
        print("Please run this script from the project root directory")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("\n🚀 Starting the application...")
    print("📱 The app will open in your default web browser")
    print("🔄 To stop the app, press Ctrl+C in this terminal")
    print("\n" + "=" * 50)
    
    try:
        # Run the Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running the application: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
