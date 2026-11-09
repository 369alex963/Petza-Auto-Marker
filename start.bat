@echo off
title Petza Auto-Marker
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First run: installing. This takes a while once.
  call "%~dp0setup.bat" || goto :fail
)
".venv\Scripts\python.exe" -m petza
if errorlevel 1 goto :fail
exit /b 0
:fail
echo.
echo Something went wrong. Read the message above.
pause
exit /b 1
