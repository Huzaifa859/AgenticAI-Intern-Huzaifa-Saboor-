@echo off
cd /d "%~dp0"

REM Prefer Google Chrome when Streamlit opens the browser.
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
  set "BROWSER=C:\Program Files\Google\Chrome\Application\chrome.exe"
) else if exist "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" (
  set "BROWSER=C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
)

streamlit run app/streamlit_app.py --server.headless=false --server.fileWatcherType=none --server.runOnSave=false
