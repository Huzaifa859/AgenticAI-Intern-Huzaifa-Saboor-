@echo off
REM Launcher for Claude Desktop MCP (stdio). Keeps cwd + UTF-8 stable.
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
"C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe" -m codebase_assistant.mcp
