@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0capture_windows.ps1" %*
endlocal
