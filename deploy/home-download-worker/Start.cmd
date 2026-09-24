@echo off
cd /d "%~dp0"
if exist STOP del /q STOP
".venv\Scripts\python.exe" worker.py
pause
