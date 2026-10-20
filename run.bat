@echo off
rem FlyBrain launcher. For now it only runs the machine self-check (doctor).
rem This file must stay pure ASCII: cmd mis-parses non-ASCII text after chcp.
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 goto no_uv
uv run flybrain doctor
goto done

:no_uv
type "%~dp0scripts\no_uv.txt"

:done
echo.
pause
