@echo off
REM Create and use a dedicated venv for Gold Ornament Classifier (avoids conflicts with tensorflow/yolov7)
if not exist venv (
    python -m venv venv
    echo Created venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo.
echo Done. To run the app:
echo   venv\Scripts\activate
echo   streamlit run app.py
pause
