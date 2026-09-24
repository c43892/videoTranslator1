@echo off
cd /d "%~dp0"
type nul > STOP
echo Stop requested. Current transfers will be cancelled; the server will recover their leases.
pause
