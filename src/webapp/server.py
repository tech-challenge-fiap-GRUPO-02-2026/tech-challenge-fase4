"""Painel da equipe médica — backend HTTP só com a biblioteca padrão (mesmo padrão da Fase 3).

    py -m src.webapp.server [--port 8766]

Endpoints:
    GET  /                                 painel (static/index.html)
    GET  /api/patients                     pacientes + risco fundido
    GET  /api/patient/<id>                 avaliação multimodal completa
    POST /api/patient/<id>/analyze         reprocessa o paciente e emite alertas
    GET  /api/stream?patient_id=&speed=    Server-Sent Events: sinais vitais ao vivo + alertas
    GET  /api/alerts?n=50                  alertas recentes
    POST /api/alerts/<id>/ack              reconhece um alerta {"user": "..."}
    GET  /api/azure/status                 status dos serviços Azure configurados
    GET  /media/<audio|video|output>/...   arquivos de mídia e frames anotados
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import queue
import threading
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.alerts.alert_manager import AlertManager
from src.cloud.azure_client import AzureSettings
from src.config import AUDIO_DIR, OUTPUT_DIR, VIDEO_DIR
from src.monitoring.realtime import replay
from src.pipeline import MonitoringSystem

STATIC_DIR = Path(__file__).parent / "static"
MEDIA_ROOTS = {"audio": AUDIO_DIR, "video": VIDEO_DIR, "output": OUTPUT_DIR}


class State:
    def __init__(self, use_audioset_model: bool = False):
        self.alerts = AlertManager()
        self.system = MonitoringSystem(alert_manager=self.alerts, use_audioset_model=use_audioset_model)
        self.cache: dict[str, dict] = {}
        self.lock = threading.Lock()

    def assessment(self, pid: str, refresh: bool = False) -> dict:
        with self.lock:
            if refresh or pid not in self.cache:
                self.cache[pid] = self.system.assess(pid, emit_alerts=refresh).to_dict()
            return self.cache[pid]

    def warm_up(self) -> None:
        for pid in self.system.patients["patient_id"]:
            self.assessment(pid)


STATE: State | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "MonitorMultimodal/1.0"

    def log_message(self, fmt, *args):  # silencioso; a auditoria registra o que importa
        return

    # -- helpers --------------------------------------------------------------------------
    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self._json({"erro": "não encontrado"}, 404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):  # range requests: <video>/<audio> precisam para seek
            start_s, _, end_s = rng[6:].partition("-")
            start = int(start_s or 0)
            end = int(end_s) if end_s else len(data) - 1
            chunk = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        else:
            chunk = data
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    # -- rotas ----------------------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        qs = parse_qs(url.query)
        try:
            if not parts:
                return self._file(STATIC_DIR / "index.html")
            if parts[0] == "media" and len(parts) >= 3 and parts[1] in MEDIA_ROOTS:
                root = MEDIA_ROOTS[parts[1]].resolve()
                target = (root / "/".join(parts[2:])).resolve()
                if root not in target.parents:
                    return self._json({"erro": "caminho inválido"}, 400)
                return self._file(target)
            if parts == ["api", "patients"]:
                rows = []
                for p in STATE.system.patients.to_dict("records"):
                    a = STATE.assessment(p["patient_id"])
                    rows.append({**p, "risco": a["fusao"]["score"], "nivel": a["fusao"]["nivel"],
                                 "sindromes": [s["kind"] for s in a["fusao"]["sindromes"]]})
                return self._json(rows)
            if len(parts) == 3 and parts[:2] == ["api", "patient"]:
                return self._json(STATE.assessment(parts[2]))
            if parts == ["api", "alerts"]:
                n = int(qs.get("n", ["50"])[0])
                return self._json([a.to_dict() for a in STATE.alerts.recent(n)][::-1])
            if parts == ["api", "azure", "status"]:
                return self._json(AzureSettings.from_env().status())
            if parts == ["api", "stream"]:
                return self._stream(qs.get("patient_id", ["PAC-0004"])[0], float(qs.get("speed", ["8"])[0]))
            return self._json({"erro": "rota não encontrada"}, 404)
        except KeyError as exc:
            return self._json({"erro": str(exc)}, 404)

    def do_POST(self):  # noqa: N802
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        if len(parts) == 4 and parts[:2] == ["api", "patient"] and parts[3] == "analyze":
            return self._json(STATE.assessment(parts[2], refresh=True))
        if len(parts) == 4 and parts[:2] == ["api", "alerts"] and parts[3] == "ack":
            alert = STATE.alerts.acknowledge(parts[2], self._body().get("user", "equipe"))
            return self._json(alert.to_dict() if alert else {"erro": "alerta não encontrado"}, 200 if alert else 404)
        return self._json({"erro": "rota não encontrada"}, 404)

    def _stream(self, patient_id: str, speed: float) -> None:
        """SSE: cada amostra de vitais vira evento 'vitals'; cada alerta, evento 'alert'."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        events: queue.Queue = queue.Queue()
        closed = threading.Event()

        def send(kind: str, payload) -> None:
            try:
                self.wfile.write(f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                closed.set()

        def on_alert(alert):
            events.put(("alert", alert.to_dict()))

        session = uuid.uuid4().hex[:8]  # cada sessão de monitor é um episódio próprio (deduplicação)
        STATE.alerts.subscribe(on_alert)
        try:
            df = STATE.system.vitals_frame(patient_id)

            def on_sample(t, sample):
                send("vitals", {"t": t, **sample})
                while not events.empty():
                    send(*events.get())

            def on_finding(f):
                if f.severity == "info":
                    send("info", f.to_dict())
                else:
                    STATE.alerts.emit(f, scope=session)

            send("start", {"patient_id": patient_id, "fonte": df.attrs.get("source"), "duracao_s": len(df)})
            replay(df, patient_id, on_sample=on_sample, on_finding=on_finding, speed=speed,
                   should_stop=closed.is_set)
            while not events.empty():
                send(*events.get())
            send("end", {"patient_id": patient_id})
        finally:
            STATE.alerts.unsubscribe(on_alert)


def main() -> None:
    global STATE
    parser = argparse.ArgumentParser(description="Painel de monitoramento multimodal")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--open", action="store_true", help="abre o navegador")
    parser.add_argument("--audioset-model", action="store_true", help="usa o classificador AST (AudioSet)")
    args = parser.parse_args()

    STATE = State(use_audioset_model=args.audioset_model)
    print("Carregando modelos e analisando pacientes (uma vez)...")
    STATE.warm_up()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Painel disponível em {url}  (Ctrl+C para encerrar)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
