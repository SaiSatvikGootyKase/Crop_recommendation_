#!/usr/bin/env python3
"""
Crop Recommendation Web Application Startup Script
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please check your Python installation.")
        return False
    return True

def start_application():
    """Start the Flask application"""
    print("Starting Crop Recommendation Web Application...")
    print("🌱 The website will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the application")
    print()
    
    try:
        # Import and run the app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main function"""
    print("=" * 50)
    print("🌱 Crop Recommendation Web Application")
    print("=" * 50)
    print()
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    print()
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main()
