@echo off
setlocal

set "APP_DIR=%~dp0"
set "PORT=8502"
set "URL=http://localhost:%PORT%"
set "LOG_FILE=%TEMP%\tide_agent_streamlit.log"

cd /d "%APP_DIR%"

if not exist "%APP_DIR%tide_env\Scripts\streamlit.exe" (
  echo.
  echo Tide Agent launcher error:
  echo   Could not find "%APP_DIR%tide_env\Scripts\streamlit.exe"
  echo.
  echo Make sure the virtual environment is created on Windows.
  pause
  exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "(Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue ^| Measure-Object).Count"') do set "LISTEN_COUNT=%%I"
if not "%LISTEN_COUNT%"=="0" (
  start "" "%URL%"
  exit /b 0
)

start "" /min cmd /c ""%APP_DIR%tide_env\Scripts\streamlit.exe" run app.py --server.headless true --server.port %PORT% > "%LOG_FILE%" 2>&1"
timeout /t 2 /nobreak >nul
start "" "%URL%"

exit /b 0
