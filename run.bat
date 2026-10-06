@echo off
title ReviewLens - Movie Review Sentiment Analysis
echo ===================================================
echo Starting ReviewLens Streamlit Application...
echo ===================================================
streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Falling back to python -m streamlit...
    python -m streamlit run app.py
)
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error launching Streamlit. Please inspect error above.
    pause
)
