#!/usr/bin/env bash
# Sobe o painel de monitoramento multimodal e abre o navegador (Linux/Mac).
set -e
cd "$(dirname "$0")"
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "Ambiente virtual não encontrado. Crie com:"
  echo "   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
"$PY" -m src.webapp.server --port 8766 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT
sleep 12
URL="http://127.0.0.1:8766"
if command -v xdg-open >/dev/null; then xdg-open "$URL"; elif command -v open >/dev/null; then open "$URL"; fi
echo "Painel em $URL — Ctrl+C para encerrar"
wait $PID
