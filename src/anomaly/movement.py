"""Padrões de movimentação do paciente durante a internação (actigrafia de leito/pulseira).

Sinais: contagem de atividade a cada 5 min, saída do leito e pico de
impacto (g). Detecções:

* **Queda** — pico de impacto ≥ 2,5 g seguido de imobilidade (o padrão
  clássico "impacto + ausência de movimento" dos detectores de queda);
* **Imobilidade prolongada** — ≥ 3 h acordado sem nenhum
  reposicionamento (risco de lesão por pressão; protocolo de mudança de
  decúbito a cada 2 h);
* **Agitação noturna** — atividade entre 22h e 6h acima do percentil 99
  do próprio paciente nas noites anteriores (z-score robusto), padrão
  associado a delirium;
* **Saídas do leito à noite** em paciente com risco de queda;
* **Isolation Forest por hora** — perfil horário (média, máximo,
  variabilidade, saídas) atípico em relação ao resto da internação.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MOVEMENT_CSV
from src.models import Finding, modality_risk

STEP_MIN = 5
FALL_IMPACT_G = 2.5
IMMOBILITY_H = 3.0
NIGHT_HOURS = set(range(22, 24)) | set(range(0, 6))


def _finding(kind, severity, patient_id, message, time_ref, method, evidence) -> Finding:
    return Finding(modality="movement", kind=kind, severity=severity, message=message, patient_id=patient_id,
                   time_ref=time_ref, method=method, evidence=evidence)


def load(path=MOVEMENT_CSV) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def _fmt(ts) -> str:
    return pd.Timestamp(ts).strftime("%d/%m %H:%M")


def detect_falls(p: pd.DataFrame, patient_id: str) -> list[Finding]:
    out = []
    for i in np.where(p["impacto_g"].to_numpy() >= FALL_IMPACT_G)[0]:
        after = p["atividade"].iloc[i + 1:i + 7].to_numpy()
        if after.size and after.mean() < 5:
            ts = p["timestamp"].iloc[i]
            out.append(_finding("queda_leito", "critico", patient_id,
                f"Provável QUEDA às {_fmt(ts)}: impacto de {p['impacto_g'].iloc[i]:.1f} g seguido de "
                f"{len(after) * STEP_MIN} min de imobilidade", ts.isoformat(),
                "regra: impacto ≥ 2,5 g + imobilidade subsequente",
                {"impacto_g": float(p["impacto_g"].iloc[i]), "atividade_pos": round(float(after.mean()), 1)}))
    return out


def detect_immobility(p: pd.DataFrame, patient_id: str) -> list[Finding]:
    """Janelas diurnas sem nenhum pico de reposicionamento (atividade > 30)."""
    hours = p["timestamp"].dt.hour
    awake = (hours >= 7) & (hours < 22)
    still = (p["atividade"] < 30) & awake
    out, run_start = [], None
    vals = still.to_numpy()
    for i, v in enumerate(np.append(vals, False)):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            dur_h = (i - run_start) * STEP_MIN / 60
            if dur_h >= IMMOBILITY_H:
                ts = p["timestamp"].iloc[run_start]
                out.append(_finding("imobilidade_prolongada", "alto", patient_id,
                    f"{dur_h:.1f} h sem reposicionamento no leito a partir de {_fmt(ts)} (meta: mudança de decúbito a cada 2 h)",
                    ts.isoformat(), "regra: ausência de picos de atividade em período diurno",
                    {"duracao_h": round(dur_h, 1)}))
            run_start = None
    return out


def detect_night_agitation(p: pd.DataFrame, patient_id: str) -> list[Finding]:
    night = p[p["timestamp"].dt.hour.isin(NIGHT_HOURS)].copy()
    if night.empty:
        return []
    # "noite" = data do início do período noturno (madrugada pertence à noite anterior)
    night["noite"] = (night["timestamp"] - pd.Timedelta(hours=6)).dt.date
    nights = sorted(night["noite"].unique())
    out = []
    for i, n in enumerate(nights):
        cur = night[night["noite"] == n]
        ref = night[night["noite"] != n]["atividade"]
        if ref.empty:
            continue
        med = ref.median()
        mad = (ref - med).abs().median() * 1.4826 + 1e-6
        z = ((cur["atividade"] - med) / mad).to_numpy()
        # agitação = trecho CONTÍNUO (tolerando 1 amostra de pausa) — picos isolados de
        # reposicionamento pela enfermagem não contam
        best, start, gap = (0, 0, 0), None, 0
        for j, high in enumerate(np.append(z > 4, False)):
            if high:
                start = j if start is None else start
                gap = 0
            elif start is not None:
                gap += 1
                if gap > 1:
                    end = j - gap
                    if end - start + 1 > best[0]:
                        best = (end - start + 1, start, end)
                    start, gap = None, 0
        length, a, b = best
        if length >= 6:  # ≥ 30 min
            seg = cur.iloc[a:b + 1]
            ts = seg["timestamp"].iloc[0]
            exits = int(seg["saida_leito"].sum())
            out.append(_finding("agitacao_noturna", "alto", patient_id,
                f"Agitação noturna: {length * STEP_MIN} min contínuos de atividade atípica a partir de {_fmt(ts)}"
                + (f", {exits} saída(s) do leito" if exits else "") + " — avaliar delirium",
                ts.isoformat(), "z-score robusto vs. outras noites do paciente",
                {"minutos": length * STEP_MIN, "fim": seg["timestamp"].iloc[-1].isoformat(),
                 "z_max": round(float(z[a:b + 1].max()), 1), "saidas_leito": exits}))
    return out


def hourly_features(p: pd.DataFrame) -> pd.DataFrame:
    g = p.set_index("timestamp").resample("1h")
    return pd.DataFrame({
        "media": g["atividade"].mean(), "maximo": g["atividade"].max(),
        "desvio": g["atividade"].std(), "saidas": g["saida_leito"].sum(), "impacto_max": g["impacto_g"].max(),
    }).fillna(0)


def _hour_matrix(feats: pd.DataFrame) -> np.ndarray:
    hour = feats.index.hour.to_numpy()
    return np.column_stack([feats.to_numpy(), np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)])


def train_population_model(df: pd.DataFrame, contamination: float = 0.01):
    """Isolation Forest sobre os perfis horários de TODOS os pacientes internados."""
    from sklearn.ensemble import IsolationForest

    X = np.vstack([_hour_matrix(hourly_features(p.sort_values("timestamp"))) for _, p in df.groupby("patient_id")])
    return IsolationForest(n_estimators=300, contamination=contamination, random_state=0).fit(X)


def detect_isolation_forest(p: pd.DataFrame, patient_id: str, model) -> list[Finding]:
    feats = hourly_features(p)
    if feats.empty:
        return []
    flags = model.predict(_hour_matrix(feats)) == -1
    out = []
    for ts in feats.index[flags]:
        row = feats.loc[ts]
        out.append(_finding("perfil_horario_atipico", "moderado", patient_id,
            f"Perfil de movimentação atípico às {_fmt(ts)} (média {row['media']:.0f}, máx {row['maximo']:.0f}, "
            f"{int(row['saidas'])} saída(s) do leito)", ts.isoformat(), "Isolation Forest sobre perfis horários",
            {k: round(float(v), 2) for k, v in row.items()}))
    return out


def detect(df: pd.DataFrame, patient_id: str, model=None, use_ml: bool = True) -> list[Finding]:
    p = df[df["patient_id"] == patient_id].sort_values("timestamp").reset_index(drop=True)
    if p.empty:
        return []
    findings = detect_falls(p, patient_id) + detect_immobility(p, patient_id) + detect_night_agitation(p, patient_id)
    if use_ml:
        model = model if model is not None else train_population_model(df)
        # só reporta horas atípicas que não estejam já explicadas por um achado de regra
        covered = set()
        for f in findings:
            start = pd.Timestamp(f.time_ref).floor("1h")
            end = pd.Timestamp(f.evidence.get("fim", f.time_ref)).floor("1h")
            covered |= set(pd.date_range(start, end, freq="1h"))
        findings += [f for f in detect_isolation_forest(p, patient_id, model)
                     if pd.Timestamp(f.time_ref).floor("1h") not in covered]
    return sorted(findings, key=lambda f: f.time_ref or "")


def summary(df: pd.DataFrame, patient_id: str, findings: list[Finding]) -> dict:
    p = df[df["patient_id"] == patient_id]
    return {
        "periodo": [str(p["timestamp"].min()), str(p["timestamp"].max())] if not p.empty else [],
        "atividade_media": round(float(p["atividade"].mean()), 1) if not p.empty else 0.0,
        "saidas_leito": int(p["saida_leito"].sum()) if not p.empty else 0,
        "n_achados": len(findings),
        "risco_movimentacao": modality_risk(findings),
    }
