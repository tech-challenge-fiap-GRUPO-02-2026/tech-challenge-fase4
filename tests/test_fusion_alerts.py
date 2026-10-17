import json

from src.alerts.alert_manager import AlertManager
from src.fusion.multimodal_fusion import fuse
from src.models import Finding
from src.safety.audit_logger import AuditLogger


def F(modality, kind, severity="alto", pid="P"):
    return Finding(modality, kind, severity, f"{kind} detectado", patient_id=pid)


def test_stroke_syndrome_needs_two_modalities():
    only_audio = fuse("P", {"audio": [F("audio", "voz_disartria")]})
    assert not only_audio.syndromes
    both = fuse("P", {"audio": [F("audio", "voz_disartria")], "video": [F("video", "assimetria_membros_superiores")]})
    assert [s.kind for s in both.syndromes] == ["suspeita_avc"]
    assert both.level == "critico"


def test_fusion_never_dilutes_critical():
    r = fuse("P", {"vitals": [F("vitals", "spo2_baixo_regra", "critico")]})
    assert r.score >= 0.95 and r.level == "critico"
    assert fuse("P", {}).score == 0.0


def test_dedup_and_min_severity(tmp_path):
    am = AlertManager(persist=False, min_severity="alto")
    assert am.emit(F("vitals", "x", "moderado")) is None
    first = am.emit(F("vitals", "spo2", "critico"), now=1000)
    dup = am.emit(F("vitals", "spo2", "critico"), now=1100)
    later = am.emit(F("vitals", "spo2", "critico"), now=1000 + 400)
    assert first and first.priority == "P1" and dup is None and later is not None


def test_subscribers_ack_and_audit(tmp_path):
    audit = AuditLogger(tmp_path / "audit.jsonl")
    am = AlertManager(persist=True, audit=audit)
    got = []
    am.subscribe(got.append)
    a = am.emit(F("video", "queda_detectada", "critico"))
    assert got == [a]
    am.acknowledge(a.alert_id, "dra. teste")
    assert am.recent()[0].status == "reconhecido"
    types = [json.loads(line)["type"] for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert types == ["alert_emitted", "alert_acknowledged"]


def test_webhook_failure_does_not_break_alert():
    am = AlertManager(persist=False, webhook_url="http://127.0.0.1:9/nao-existe")
    a = am.emit(F("vitals", "hr", "critico"))
    assert a is not None and am.webhook_errors
