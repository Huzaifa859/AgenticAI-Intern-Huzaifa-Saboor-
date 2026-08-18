@echo off
REM Optional manual start of the warm Streamlit worker (localhost:8765).
REM Streamlit also auto-spawns this on first Run when none is running.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app\worker_server.py %*
) else (
  python app\worker_server.py %*
)
