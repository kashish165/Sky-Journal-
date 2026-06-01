@echo off
title Sky Journal
echo.
echo  ╔══════════════════════════════════════╗
echo  ║        SKY JOURNAL — starting...    ║
echo  ╚══════════════════════════════════════╝
echo.

:: Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

:: Run the app — it auto-installs all dependencies itself
python "%~dp0sky_journal.py"

if errorlevel 1 (
    echo.
    echo [!] Something went wrong. See error above.
    pause
)
