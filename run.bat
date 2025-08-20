@echo off
echo Starting Crop Recommendation Web Application...
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting the application...
echo The website will be available at: http://localhost:5000
echo Press Ctrl+C to stop the application
echo.
python app.py
pause
