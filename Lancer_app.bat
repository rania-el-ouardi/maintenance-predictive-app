@echo off
cd /d "%~dp0"
echo Lancement de l'application Maintenance Predictive...
echo.
python -m streamlit run app.py
pause
