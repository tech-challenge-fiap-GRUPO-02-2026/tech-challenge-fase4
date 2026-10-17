"""Caminhos e parâmetros centrais do sistema de monitoramento multimodal.

Tudo que é "número mágico" clínico (limiares de sinais vitais, pesos da
fusão, janelas de detecção) fica concentrado aqui, para que a equipe
médica possa revisar e ajustar sem caçar valores espalhados no código.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PHYSIONET_DIR = RAW_DIR / "physionet" / "bidmc"
AUDIOSET_DIR = RAW_DIR / "audioset"
AUDIO_DIR = DATA_DIR / "audio"
VIDEO_DIR = DATA_DIR / "video"
CLINICAL_DIR = DATA_DIR / "clinical"

REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_DIR = REPORTS_DIR / "output"
LOGS_DIR = BASE_DIR / "logs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

PATIENTS_CSV = CLINICAL_DIR / "pacientes.csv"
PRESCRIPTIONS_CSV = CLINICAL_DIR / "prescricoes.csv"
MOVEMENT_CSV = CLINICAL_DIR / "movimentacao.csv"
BLOOD_PRESSURE_CSV = CLINICAL_DIR / "pressao_arterial.csv"

# ---------------------------------------------------------------------------
# Sinais vitais — limiares inspirados no NEWS2 (Royal College of Physicians)
# (limite_inferior_critico, limite_inferior_alerta, limite_superior_alerta, limite_superior_critico)
# ---------------------------------------------------------------------------
VITAL_THRESHOLDS: dict[str, tuple[float, float, float, float]] = {
    "HR": (40, 50, 110, 130),       # batimentos por minuto
    "SpO2": (88, 92, 101, 101),     # % — não existe "alto demais" clínico
    "RESP": (8, 11, 21, 25),        # incursões por minuto
    "SBP": (90, 100, 180, 220),     # pressão sistólica mmHg
    "DBP": (40, 50, 100, 120),      # pressão diastólica mmHg
}
VITAL_LABELS = {
    "HR": "Frequência cardíaca",
    "SpO2": "Saturação de O₂",
    "RESP": "Frequência respiratória",
    "SBP": "Pressão sistólica",
    "DBP": "Pressão diastólica",
}
VITAL_UNITS = {"HR": "bpm", "SpO2": "%", "RESP": "irpm", "SBP": "mmHg", "DBP": "mmHg"}

# Persistência mínima (em segundos) para um desvio virar alerta —
# evita alarmes falsos por artefato de sensor (dedo solto no oxímetro etc.)
VITAL_PERSISTENCE_S = 10
ROLLING_WINDOW_S = 60
ROBUST_Z_THRESHOLD = 4.0

# ---------------------------------------------------------------------------
# Fusão multimodal — pesos por modalidade (somam 1.0)
# ---------------------------------------------------------------------------
FUSION_WEIGHTS = {
    "vitals": 0.30,
    "audio": 0.20,
    "video": 0.15,
    "prescriptions": 0.15,
    "movement": 0.20,
}
RISK_LEVELS = [(0.75, "critico"), (0.5, "alto"), (0.25, "moderado"), (0.0, "baixo")]

# ---------------------------------------------------------------------------
# Alertas
# ---------------------------------------------------------------------------
ALERT_DEDUP_WINDOW_S = 300
ALERT_WEBHOOK_ENV = "ALERT_WEBHOOK_URL"  # Teams/Slack/Logic Apps (opcional)


def risk_level(score: float) -> str:
    for threshold, label in RISK_LEVELS:
        if score >= threshold:
            return label
    return "baixo"
