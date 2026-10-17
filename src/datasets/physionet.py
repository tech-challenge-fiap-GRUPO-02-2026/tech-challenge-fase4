"""PhysioNet — BIDMC PPG and Respiration Dataset (v1.0.0, acesso aberto).

53 registros de 8 minutos de pacientes críticos do Beth Israel Deaconess
Medical Center, com FC, pulso, frequência respiratória e SpO₂ a 1 Hz
(arquivos ``bidmc_XX_Numerics.csv``). É o dataset de sinais vitais reais
usado na detecção de anomalias — sugerido no enunciado (physionet.org).

Referência: Pimentel, M.A.F. et al. "Towards a Robust Estimation of
Respiratory Rate from Pulse Oximeters", IEEE TBME, 2016.
https://physionet.org/content/bidmc/1.0.0/
"""
from __future__ import annotations

import urllib.request

import pandas as pd

from src.config import BLOOD_PRESSURE_CSV, PHYSIONET_DIR

BASE_URL = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/bidmc_{n}_Numerics.csv"
N_RECORDS = 53


def record_path(record: str):
    return PHYSIONET_DIR / f"bidmc_{int(record):02d}_Numerics.csv"


def download(n: int = N_RECORDS, overwrite: bool = False) -> int:
    PHYSIONET_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for i in range(1, n + 1):
        path = record_path(str(i))
        if path.exists() and not overwrite:
            continue
        urllib.request.urlretrieve(BASE_URL.format(n=f"{i:02d}"), path)
        downloaded += 1
    return downloaded


def load_record(record: str) -> pd.DataFrame:
    """Carrega um registro BIDMC como DataFrame com colunas time_s, HR, PULSE, RESP, SpO2.

    Lacunas curtas (perda de sinal do sensor) são interpoladas; o número de
    amostras faltantes fica em ``df.attrs["missing"]`` para auditoria.
    """
    df = pd.read_csv(record_path(record))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Time [s]": "time_s"})
    df.attrs["missing"] = int(df.isna().sum().sum())
    cols = ["HR", "PULSE", "RESP", "SpO2"]
    df[cols] = df[cols].astype(float).interpolate(limit_direction="both")
    df.attrs["source"] = f"PhysioNet BIDMC registro {int(record):02d}"
    return df


def load_patient_vitals(patient_id: str, record: str) -> pd.DataFrame:
    """Sinais vitais completos do paciente: BIDMC real + PA arterial sintética alinhada no tempo."""
    df = load_record(record)
    if BLOOD_PRESSURE_CSV.exists():
        bp = pd.read_csv(BLOOD_PRESSURE_CSV)
        bp = bp[bp["patient_id"] == patient_id].drop(columns="patient_id")
        if not bp.empty:
            attrs = dict(df.attrs)
            df = df.merge(bp, on="time_s", how="left")
            df[["SBP", "DBP"]] = df[["SBP", "DBP"]].interpolate(limit_direction="both")
            df.attrs.update(attrs)
    return df


def load_population(exclude: set[str] | None = None) -> list[pd.DataFrame]:
    """Todos os registros disponíveis (exceto os dos pacientes monitorados) — base do modelo populacional."""
    exclude = {f"{int(r):02d}" for r in (exclude or set())}
    frames = []
    for path in sorted(PHYSIONET_DIR.glob("bidmc_*_Numerics.csv")):
        rec = path.stem.split("_")[1]
        if rec in exclude:
            continue
        frames.append(load_record(rec))
    return frames
