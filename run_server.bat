@echo off
REM Have Hit API server. Double-click this, or run it from any shell.
REM
REM PYTHONUNBUFFERED is set HERE as well as inside run_server.py, and that is
REM not redundant. The interpreter reads this variable once, at startup, to
REM decide how stdout is buffered -- by the time Python code runs it is too
REM late for the env var to change anything about the stream that already
REM exists. Setting it here is what makes the CURRENT process unbuffered;
REM run_server.py's reconfigure() is the fallback for when this file was
REM bypassed.
setlocal
set PYTHONUNBUFFERED=1

cd /d "%~dp0"

REM Prefer the project venv. Falling through to whatever `python` resolves to
REM is a real hazard on this machine -- torch, the V-JEPA hub cache and the
REM SAM checkpoint all live in the venv, and the system interpreter would get
REM as far as `import torch` before saying so.
set PY=%~dp0venv\Scripts\python.exe
if not exist "%PY%" (
  echo [warn] venv not found at %PY%, falling back to PATH python
  set PY=python
)

"%PY%" run_server.py %*
set CODE=%ERRORLEVEL%

REM Hold the window open on failure so a double-click user can read the error
REM instead of watching it flash past.
if not "%CODE%"=="0" (
  echo.
  echo Server exited with code %CODE%.
  pause
)
endlocal & exit /b %CODE%
