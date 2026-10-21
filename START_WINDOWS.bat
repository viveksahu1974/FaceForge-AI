@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    py -3.11 -m venv .venv
    if ERRORLEVEL 1 (
        echo Python 3.11 not available. Trying Python 3.10...
        py -3.10 -m venv .venv
        if ERRORLEVEL 1 (
            echo Install Python 3.10 or 3.11.
            pause
            exit /b 1
        )
    )
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if ERRORLEVEL 1 goto failure
.venv\Scripts\python.exe -m pip install -e .
if ERRORLEVEL 1 goto failure
set NO_PROXY=localhost,127.0.0.1,::1
set no_proxy=localhost,127.0.0.1,::1
echo.
echo Opening FaceForge AI on http://127.0.0.1:7860
.venv\Scripts\python.exe app.py
pause
exit /b 0
:failure
echo Installation failed. Please check the output above.
pause
exit /b 1
