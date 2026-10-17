import numpy as np
import pandas as pd

from src.anomaly import vitals
from src.datasets import physionet
from src.monitoring.realtime import OnlineVitalsMonitor, replay


def _frame(n=300, **overrides):
    t = np.arange(n)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"time_s": t, "HR": 80 + rng.normal(0, 1, n), "SpO2": 97 + rng.normal(0, 0.3, n),
                       "RESP": 16 + rng.normal(0, 0.5, n)})
    for col, (a, b, v) in overrides.items():
        df.loc[(df.time_s >= a) & (df.time_s < b), col] = v
    return df


def test_physionet_record_loads():
    df = physionet.load_record("32")
    assert {"time_s", "HR", "SpO2", "RESP"} <= set(df.columns)
    assert len(df) == 481 and df["SpO2"].max() < 90  # hipoxemia real do BIDMC 32


def test_stable_signal_has_no_findings():
    findings, _ = vitals.detect(_frame())
    assert findings == []


def test_rule_requires_persistence():
    short, _ = vitals.detect(_frame(SpO2=(100, 105, 84)))      # 5 s — artefato
    sustained, _ = vitals.detect(_frame(SpO2=(100, 140, 84)))  # 40 s
    assert not any(f.method == "regra_clinica_news2" for f in short)
    crit = [f for f in sustained if f.method == "regra_clinica_news2"]
    assert crit and crit[0].severity == "critico" and crit[0].kind.startswith("spo2_baixo")


def test_zscore_catches_abrupt_change_inside_normal_range():
    findings, _ = vitals.detect(_frame(HR=(150, 170, 104)))   # 80 → 104 bpm: ainda < 110
    assert any(f.method.startswith("zscore") and f.evidence["sinal"] == "HR" for f in findings)


def test_consolidation_merges_fragments():
    df = _frame(n=400)
    df.loc[(df.time_s >= 100) & (df.time_s < 300), "RESP"] = np.where(np.arange(200) % 20 < 15, 26, 24)
    findings, _ = vitals.detect(df)
    rules = [f for f in findings if f.method == "regra_clinica_news2" and f.evidence["sinal"] == "RESP"]
    assert len(rules) == 1


def test_isolation_forest_flags_hypoxemic_record():
    pop = vitals.train_population_model([_frame(n=480), _frame(n=480, HR=(0, 480, 95))])
    hyp = physionet.load_record("32")
    assert vitals.isolation_forest(hyp, pop)


def test_online_monitor_latency_and_rearm():
    df = _frame(SpO2=(100, 140, 84))
    findings, mon = replay(df, "T")
    fired = [f for f in findings if f.kind == "spo2_baixo_tempo_real"]
    assert len(fired) == 1
    assert 9 <= mon.latencies[0] <= 11  # persistência de 10 s
    assert any(f.kind == "spo2_normalizado" for f in findings)


def test_online_monitor_escalates_to_critical():
    m = OnlineVitalsMonitor("T", persistence_s=3)
    out = []
    for t in range(10):
        out += m.update(t, {"SpO2": 90.0})          # alto
    for t in range(10, 15):
        out += m.update(t, {"SpO2": 84.0})          # crítico
    sev = [f.severity for f in out if f.kind.startswith("spo2_baixo")]
    assert sev == ["alto", "critico"]
