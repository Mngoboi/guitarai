@echo off
cd /d "%~dp0"
echo Abriendo GuitarAI...
if exist venv\Scripts\python.exe (
  venv\Scripts\python.exe app_nativa.py
) else (
  python app_nativa.py
)
pause
