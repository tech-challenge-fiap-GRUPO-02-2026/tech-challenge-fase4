from datetime import datetime

import numpy as np
import pandas as pd

from src.anomaly import movement, prescriptions


def _rx(rows):
    return pd.DataFrame(rows, columns=["patient_id", "timestamp", "medicamento", "classe", "dose_mg",
                                       "intervalo_h", "via", "acao", "prescritor"]).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"]))


def test_allergy_conflict_and_dose_jump():
    df = _rx([
        ("P", "2026-10-10T08:00", "piperacilina_tazobactam", "antibiotico", 4500, 6, "EV", "iniciar", "A"),
        ("P", "2026-10-10T08:00", "vancomicina", "antibiotico", 1000, 12, "EV", "iniciar", "A"),
        ("P", "2026-10-11T08:00", "vancomicina", "antibiotico", 3000, 12, "EV", "ajustar", "B"),
    ])
    kinds = [f.kind for f in prescriptions.detect(df, "P", "penicilina")]
    assert "conflito_alergia" in kinds and "salto_de_dose" in kinds and "dose_acima_maxima" in kinds


def test_usual_titration_is_not_flagged():
    df = _rx([
        ("P", "2026-10-10T08:00", "atorvastatina", "estatina", 40, 24, "VO", "iniciar", "A"),
        ("P", "2026-10-11T08:00", "atorvastatina", "estatina", 80, 24, "VO", "ajustar", "A"),
    ])
    assert prescriptions.detect(df, "P") == []


def test_anticoagulant_suspension_and_duplicity():
    df = _rx([
        ("P", "2026-10-10T08:00", "heparina", "anticoagulante", 25000, 24, "EV", "iniciar", "A"),
        ("P", "2026-10-10T20:00", "enoxaparina", "anticoagulante", 80, 12, "SC", "iniciar", "B"),
        ("P", "2026-10-11T08:00", "heparina", "anticoagulante", 0, 0, "EV", "suspender", "B"),
        ("P", "2026-10-11T08:00", "enoxaparina", "anticoagulante", 0, 0, "SC", "suspender", "B"),
    ])
    kinds = [f.kind for f in prescriptions.detect(df, "P")]
    assert "duplicidade_terapeutica" in kinds
    assert kinds.count("suspensao_anticoagulante_sem_substituto") == 1


def _movement(n_hours=48, **events):
    rng = np.random.default_rng(0)
    ts = pd.date_range(datetime(2026, 10, 10, 8), periods=n_hours * 12, freq="5min")
    hours = ts.hour
    act = np.where((hours >= 7) & (hours < 22), rng.gamma(3, 12, len(ts)), rng.gamma(1.2, 4, len(ts)))
    act[::24] += 80
    df = pd.DataFrame({"patient_id": "P", "timestamp": ts, "atividade": act, "saida_leito": 0,
                       "impacto_g": 1.0})
    if "fall_at" in events:
        i = events["fall_at"]
        df.loc[i, "impacto_g"] = 3.2
        df.loc[i + 1:i + 7, "atividade"] = 0.5
    return df


def test_fall_detection_movement():
    kinds = [f.kind for f in movement.detect(_movement(fall_at=300), "P", use_ml=False)]
    assert "queda_leito" in kinds
    assert movement.detect(_movement(), "P", use_ml=False) == []
