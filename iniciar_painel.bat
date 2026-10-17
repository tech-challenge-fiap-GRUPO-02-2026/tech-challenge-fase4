@echo off
REM Sobe o painel de monitoramento multimodal e abre o navegador.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Ambiente virtual nao encontrado. Crie com:
  echo    py -m venv .venv
  echo    .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
start "Painel Multimodal - backend" cmd /k ".venv\Scripts\python.exe -m src.webapp.server --port 8766"
timeout /t 12 /nobreak >nul
start "" http://127.0.0.1:8766
