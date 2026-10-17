import pytest

from src.alerts.alert_manager import AlertManager
from src.pipeline import MonitoringSystem


@pytest.fixture(scope="module")
def system():
    return MonitoringSystem(alert_manager=AlertManager(persist=False), prefer_azure=False)


def test_control_patient_is_low_risk(system):
    a = system.assess("PAC-0001", emit_alerts=False)
    assert a.fusion["nivel"] == "baixo"
    assert not a.all_findings()


@pytest.mark.parametrize("pid,syndrome", [
    ("PAC-0002", "insuficiencia_respiratoria"),
    ("PAC-0003", "suspeita_avc"),
    ("PAC-0004", "possivel_choque"),
    ("PAC-0005", "queda_confirmada"),
    ("PAC-0006", "risco_seguranca_medicamentosa"),
])
def test_expected_syndromes(system, pid, syndrome):
    a = system.assess(pid, emit_alerts=False)
    assert syndrome in [s["kind"] for s in a.fusion["sindromes"]]
    assert a.fusion["nivel"] == "critico"


def test_alerts_emitted_and_routed(system):
    a = system.assess("PAC-0005")
    assert a.alerts and a.alerts[0]["priority"] == "P1"
