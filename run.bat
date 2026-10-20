@echo off
rem FlyBrain launcher: starts the server and opens the browser with the 3D neuron cloud.
rem This file must stay pure ASCII: cmd mis-parses non-ASCII text after chcp.
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 goto no_uv
uv run flybrain serve
goto done

:no_uv
type "%~dp0scripts\no_uv.txt"

:done
echo.
pause
