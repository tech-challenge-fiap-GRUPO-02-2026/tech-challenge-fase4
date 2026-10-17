"""Trilha de auditoria (JSON Lines) de toda análise e todo alerta emitido.

Mesmo padrão da Fase 3: cada evento tem ``event_id`` único e timestamp
UTC, e registra qual modelo/provider produziu o resultado (incluindo se
foi Azure ou fallback local) — essencial para rastrear por que um alerta
foi ou não disparado.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import LOGS_DIR

AUDIT_PATH = LOGS_DIR / "audit_log.jsonl"
_lock = threading.Lock()


class AuditLogger:
    def __init__(self, path: Path = AUDIT_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, patient_id: str | None = None, **payload) -> str:
        event_id = str(uuid.uuid4())
        record = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "patient_id": patient_id,
            **payload,
        }
        with _lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return event_id

    def read_recent(self, n: int = 20, event_type: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if event_type:
            records = [r for r in records if r.get("type") == event_type]
        return records[-n:]
