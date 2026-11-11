@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Starting TingMiaoShuo local service...

rem ── Python 解释器：环境变量 > 项目内 .venv > 本机托管版 > PATH ──
if not defined TMS_PYTHON if exist "%~dp0.venv\Scripts\python.exe" set "TMS_PYTHON=%~dp0.venv\Scripts\python.exe"
if not defined TMS_PYTHON if exist "%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe" set "TMS_PYTHON=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if not defined TMS_PYTHON set "TMS_PYTHON=python"

rem ── Node：PATH 里有就用，否则退回本机托管版 ──
set "NODE=node"
where node >nul 2>nul
if errorlevel 1 set "NODE=%USERPROFILE%\.workbuddy\binaries\node\versions\22.22.2-6\node.exe"

echo URL: http://127.0.0.1:8790
echo (Close this window to stop the service)
start "" cmd /c "ping -n 3 127.0.0.1 >nul & start http://127.0.0.1:8790"
"%NODE%" server.js
