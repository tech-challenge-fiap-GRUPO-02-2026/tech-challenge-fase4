"""Alertas para a equipe médica: priorização, deduplicação, roteamento e entrega.

Fluxo:  Finding ──► Alert (prioridade + destinatário) ──► canais
                                   │
                                   └── deduplicação (mesmo paciente + mesmo tipo em 5 min)

* **Prioridade / roteamento** (inspirado em protocolos de *rapid response*):
  - ``critico`` → P1, *time de resposta rápida* + médico plantonista, imediato;
  - ``alto``    → P2, médico assistente + enfermagem do setor, ≤ 15 min;
  - ``moderado``→ P3, painel da enfermagem (não "toca"; aparece no painel);
* **Canais**: console, arquivo ``logs/alerts.jsonl``, assinantes em memória
  (painel web via Server-Sent Events) e **webhook** opcional
  (``ALERT_WEBHOOK_URL`` — Microsoft Teams / Azure Logic Apps / Slack),
  que é como o alerta sai do hospital para o celular do plantonista;
* Todo alerta é gravado na trilha de auditoria e pode ser **reconhecido**
  (ack) por um profissional, fechando o ciclo.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable

from src.config import ALERT_DEDUP_WINDOW_S, ALERT_WEBHOOK_ENV, LOGS_DIR
from src.models import SEVERITY_ORDER, Finding
from src.safety.audit_logger import AuditLogger

ALERTS_PATH = LOGS_DIR / "alerts.jsonl"

ROUTING = {
    "critico": {"prioridade": "P1", "destino": "Time de resposta rápida + médico plantonista", "prazo_min": 0},
    "alto": {"prioridade": "P2", "destino": "Médico assistente + enfermagem do setor", "prazo_min": 15},
    "moderado": {"prioridade": "P3", "destino": "Painel da enfermagem", "prazo_min": 60},
}
MODALITY_ICON = {"vitals": "❤️", "audio": "🎙️", "video": "🎥", "prescriptions": "💊", "movement": "🛏️", "fusion": "🧩"}


@dataclass
class Alert:
    alert_id: str
    created_at: str
    patient_id: str | None
    severity: str
    priority: str
    route_to: str
    deadline_min: int
    modality: str
    kind: str
    title: str
    message: str
    time_ref: str | None
    method: str | None
    evidence: dict = field(default_factory=dict)
    status: str = "novo"
    acknowledged_by: str | None = None
    acknowledged_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class AlertManager:
    def __init__(self, min_severity: str = "moderado", dedup_window_s: int = ALERT_DEDUP_WINDOW_S,
                 console: bool = False, persist: bool = True, audit: AuditLogger | None = None,
                 webhook_url: str | None = None):
        self.min_severity = min_severity
        self.dedup_window_s = dedup_window_s
        self.console = console
        self.persist = persist
        self.audit = audit or AuditLogger()
        self.webhook_url = webhook_url if webhook_url is not None else os.getenv(ALERT_WEBHOOK_ENV)
        self.alerts: deque[Alert] = deque(maxlen=1000)
        self._last_sent: dict[tuple, float] = {}
        self._subscribers: list[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        self.webhook_errors: list[str] = []

    # -- assinatura (painel web / SSE) ---------------------------------------------------
    def subscribe(self, callback: Callable[[Alert], None]) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Alert], None]) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    # -- emissão ----------------------------------------------------------------------------
    def _title(self, f: Finding) -> str:
        icon = MODALITY_ICON.get(f.modality, "⚠️")
        return f"{icon} [{ROUTING[f.severity]['prioridade']}] {f.patient_id or ''} — {f.kind.replace('_', ' ')}"

    def emit(self, finding: Finding, now: float | None = None, scope: str | None = None) -> Alert | None:
        if finding.severity not in ROUTING or SEVERITY_ORDER[finding.severity] < SEVERITY_ORDER[self.min_severity]:
            return None
        now = time.time() if now is None else now
        key = (scope, finding.patient_id, finding.kind)
        with self._lock:
            last = self._last_sent.get(key)
            if last is not None and now - last < self.dedup_window_s:
                return None
            self._last_sent[key] = now
            route = ROUTING[finding.severity]
            alert = Alert(
                alert_id=str(uuid.uuid4())[:8], created_at=datetime.now(timezone.utc).isoformat(),
                patient_id=finding.patient_id, severity=finding.severity, priority=route["prioridade"],
                route_to=route["destino"], deadline_min=route["prazo_min"], modality=finding.modality,
                kind=finding.kind, title=self._title(finding), message=finding.message,
                time_ref=finding.time_ref, method=finding.method, evidence=finding.evidence,
            )
            self.alerts.append(alert)
        self._dispatch(alert)
        return alert

    def emit_all(self, findings: list[Finding]) -> list[Alert]:
        ordered = sorted(findings, key=lambda f: -SEVERITY_ORDER.get(f.severity, 0))
        return [a for a in (self.emit(f) for f in ordered) if a is not None]

    def _dispatch(self, alert: Alert) -> None:
        if self.console:
            color = {"critico": "\033[91m", "alto": "\033[93m", "moderado": "\033[96m"}.get(alert.severity, "")
            print(f"{color}{alert.title}\033[0m\n    {alert.message}\n    → {alert.route_to}")
        if self.persist:
            ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with ALERTS_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert.to_dict(), ensure_ascii=False, default=str) + "\n")
            self.audit.log("alert_emitted", alert.patient_id, alert_id=alert.alert_id, severity=alert.severity,
                           kind=alert.kind, modality=alert.modality, route_to=alert.route_to)
        if self.webhook_url and alert.severity in ("critico", "alto"):
            self._post_webhook(alert)
        for cb in list(self._subscribers):
            try:
                cb(alert)
            except Exception:  # um assinante com defeito não pode derrubar o alerta
                self.unsubscribe(cb)

    def _post_webhook(self, alert: Alert) -> None:
        """Payload compatível com Incoming Webhook do Teams (MessageCard) e com Logic Apps (JSON livre)."""
        body = {
            "@type": "MessageCard", "@context": "https://schema.org/extensions",
            "themeColor": "D13438" if alert.severity == "critico" else "FFB900",
            "summary": alert.title, "title": alert.title, "text": alert.message,
            "sections": [{"facts": [
                {"name": "Paciente", "value": alert.patient_id or "—"},
                {"name": "Prioridade", "value": alert.priority},
                {"name": "Destino", "value": alert.route_to},
                {"name": "Origem", "value": f"{alert.modality} · {alert.method}"},
            ]}],
            "alert": alert.to_dict(),
        }
        req = urllib.request.Request(self.webhook_url, data=json.dumps(body, default=str).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as exc:  # falha de webhook é registrada, nunca silencia o alerta local
            self.webhook_errors.append(str(exc))
            self.audit.log("alert_webhook_error", alert.patient_id, alert_id=alert.alert_id, error=str(exc))

    # -- ciclo de vida ----------------------------------------------------------------------
    def acknowledge(self, alert_id: str, user: str) -> Alert | None:
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = "reconhecido"
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now(timezone.utc).isoformat()
                self.audit.log("alert_acknowledged", alert.patient_id, alert_id=alert_id, user=user)
                return alert
        return None

    def recent(self, n: int = 50, patient_id: str | None = None) -> list[Alert]:
        items = [a for a in self.alerts if patient_id is None or a.patient_id == patient_id]
        return items[-n:]
