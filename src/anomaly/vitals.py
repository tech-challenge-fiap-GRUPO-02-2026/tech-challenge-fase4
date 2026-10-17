"""Detecção de anomalias em séries temporais de sinais vitais.

Quatro camadas complementares, cada achado indica o método que o gerou:

1. **Regras clínicas** (limiares inspirados no NEWS2) com persistência
   mínima — o que o médico reconhece de imediato (SpO₂ < 88%, FC > 130);
2. **Z-score robusto em janela móvel** (mediana/MAD dos últimos 60 s) —
   mudanças abruptas em relação ao próprio paciente, mesmo dentro da
   faixa "normal" (ex.: FC que salta de 60 para 95);
3. **CUSUM** — deriva lenta e sustentada (dessaturação progressiva) que
   nenhum ponto isolado denuncia;
4. **Isolation Forest** treinado na população PhysioNet BIDMC (registros
   não monitorados) sobre janelas de 30 s (média, desvio, inclinação de
   FC/SpO₂/FR) — combinações multivariadas atípicas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    ROBUST_Z_THRESHOLD,
    ROLLING_WINDOW_S,
    VITAL_LABELS,
    VITAL_PERSISTENCE_S,
    VITAL_THRESHOLDS,
    VITAL_UNITS,
)
from src.models import Finding, modality_risk

SIGNALS = ["HR", "SpO2", "RESP", "SBP", "DBP"]
ML_SIGNALS = ["HR", "SpO2", "RESP"]
WINDOW_ML_S = 30
STEP_ML_S = 10


@dataclass
class Episode:
    signal: str
    start_s: int
    end_s: int
    extreme: float
    direction: str  # "alto" | "baixo"
    severity: str
    method: str

    @property
    def duration_s(self) -> int:
        return self.end_s - self.start_s + 1


def _episodes(mask: np.ndarray, t: np.ndarray, min_len: int, max_gap: int = 3) -> list[tuple[int, int]]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    groups, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap:
            groups.append((start, prev))
            start = i
        prev = i
    groups.append((start, prev))
    return [(int(t[a]), int(t[b])) for a, b in groups if b - a + 1 >= min_len]


# ---------------------------------------------------------------------------
# 1. Regras clínicas
# ---------------------------------------------------------------------------
def rule_based(df: pd.DataFrame, persistence_s: int = VITAL_PERSISTENCE_S) -> list[Episode]:
    eps: list[Episode] = []
    t = df["time_s"].to_numpy()
    for sig in SIGNALS:
        if sig not in df:
            continue
        x = df[sig].to_numpy(dtype=float)
        lo_crit, lo_warn, hi_warn, hi_crit = VITAL_THRESHOLDS[sig]
        for mask, direction, severity in [
            (x < lo_crit, "baixo", "critico"),
            ((x >= lo_crit) & (x < lo_warn), "baixo", "alto"),
            (x > hi_crit, "alto", "critico"),
            ((x <= hi_crit) & (x > hi_warn), "alto", "alto"),
        ]:
            for a, b in _episodes(mask, t, persistence_s):
                seg = x[(t >= a) & (t <= b)]
                extreme = seg.min() if direction == "baixo" else seg.max()
                eps.append(Episode(sig, a, b, float(extreme), direction, severity, "regra_clinica_news2"))
    return _drop_nested(eps)


def _drop_nested(eps: list[Episode]) -> list[Episode]:
    """Episódio "alto" contido num "critico" do mesmo sinal é redundante — fica só o crítico."""
    crit = [e for e in eps if e.severity == "critico"]
    out = []
    for e in eps:
        if e.severity != "critico" and any(
            c.signal == e.signal and c.direction == e.direction and c.start_s <= e.end_s and e.start_s <= c.end_s
            for c in crit
        ):
            # mantém só se for muito mais longo que o crítico sobreposto (tendência de fundo)
            overl = [c for c in crit if c.signal == e.signal and c.direction == e.direction]
            if e.duration_s <= 3 * sum(c.duration_s for c in overl):
                continue
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# 2. Z-score robusto em janela móvel
# ---------------------------------------------------------------------------
def robust_zscore(df: pd.DataFrame, window_s: int = ROLLING_WINDOW_S, threshold: float = ROBUST_Z_THRESHOLD,
                  min_len: int = 5) -> list[Episode]:
    eps: list[Episode] = []
    t = df["time_s"].to_numpy()
    for sig in SIGNALS:
        if sig not in df:
            continue
        s = df[sig].astype(float)
        med = s.rolling(window_s, min_periods=window_s // 2).median().shift(1)
        mad = (s - med).abs().rolling(window_s, min_periods=window_s // 2).median().shift(1)
        # piso do MAD: evita z infinito em sinais quase constantes (SpO₂ travada em 99%)
        floor = {"HR": 1.5, "SpO2": 0.75, "RESP": 1.5, "SBP": 2.0, "DBP": 1.5}[sig]
        z = (s - med) / (1.4826 * mad.clip(lower=floor))
        z = z.to_numpy()
        for direction, mask in [("alto", z > threshold), ("baixo", z < -threshold)]:
            for a, b in _episodes(np.nan_to_num(mask), t, min_len):
                seg = s.to_numpy()[(t >= a) & (t <= b)]
                extreme = seg.max() if direction == "alto" else seg.min()
                eps.append(Episode(sig, a, b, float(extreme), direction, "moderado", "zscore_robusto_janela_60s"))
    return eps


# ---------------------------------------------------------------------------
# 3. CUSUM — deriva sustentada
# ---------------------------------------------------------------------------
CUSUM_PARAMS = {  # sinal: (direção, folga k, limiar h, severidade) em unidades do sinal
    "SpO2": ("baixo", 1.0, 20.0, "alto"),
    "HR": ("alto", 5.0, 150.0, "moderado"),
    "RESP": ("alto", 3.0, 80.0, "moderado"),
    "SBP": ("baixo", 5.0, 150.0, "alto"),
}


def cusum(df: pd.DataFrame, baseline_s: int = 60) -> list[Episode]:
    eps: list[Episode] = []
    t = df["time_s"].to_numpy()
    for sig, (direction, k, h, severity) in CUSUM_PARAMS.items():
        if sig not in df:
            continue
        x = df[sig].to_numpy(dtype=float)
        ref = np.median(x[:baseline_s])
        s, alarm_start = 0.0, None
        for i, v in enumerate(x):
            dev = (ref - v) if direction == "baixo" else (v - ref)
            s = max(0.0, s + dev - k)
            if s > h and alarm_start is None:
                alarm_start = i
            if s == 0 and alarm_start is not None:
                eps.append(_cusum_episode(sig, x, t, alarm_start, i - 1, direction, ref, severity))
                alarm_start = None
        if alarm_start is not None:
            eps.append(_cusum_episode(sig, x, t, alarm_start, len(x) - 1, direction, ref, severity))
    return eps


def _cusum_episode(sig, x, t, a, b, direction, ref, severity) -> Episode:
    seg = x[a:b + 1]
    extreme = seg.min() if direction == "baixo" else seg.max()
    ep = Episode(sig, int(t[a]), int(t[b]), float(extreme), direction, severity, "cusum_deriva")
    ep.reference = float(ref)  # type: ignore[attr-defined]
    return ep


# ---------------------------------------------------------------------------
# 4. Isolation Forest populacional
# ---------------------------------------------------------------------------
def window_features(df: pd.DataFrame, window_s: int = WINDOW_ML_S, step_s: int = STEP_ML_S) -> tuple[np.ndarray, list[int]]:
    rows, starts = [], []
    x = df[ML_SIGNALS].to_numpy(dtype=float)
    tt = np.arange(window_s)
    for start in range(0, len(df) - window_s + 1, step_s):
        w = x[start:start + window_s]
        slope = np.polyfit(tt, w, 1)[0]
        rows.append(np.concatenate([w.mean(0), w.std(0), slope]))
        starts.append(int(df["time_s"].iloc[start]))
    return np.array(rows), starts


@dataclass
class PopulationModel:
    model: object
    threshold: float
    n_records: int
    n_windows: int
    feature_names: list[str] = field(default_factory=lambda: [
        f"{stat}_{s}" for stat in ("media", "desvio", "inclinacao") for s in ML_SIGNALS
    ])


def train_population_model(frames: list[pd.DataFrame], contamination_pct: float = 2.0, seed: int = 0) -> PopulationModel:
    from sklearn.ensemble import IsolationForest

    X = np.vstack([window_features(f)[0] for f in frames])
    model = IsolationForest(n_estimators=300, random_state=seed).fit(X)
    threshold = float(np.percentile(model.score_samples(X), contamination_pct))
    return PopulationModel(model, threshold, len(frames), len(X))


def isolation_forest(df: pd.DataFrame, pop: PopulationModel) -> list[Episode]:
    X, starts = window_features(df)
    if len(X) == 0:
        return []
    scores = pop.model.score_samples(X)
    eps = []
    flagged = [(s, sc, x) for s, sc, x in zip(starts, scores, X) if sc < pop.threshold]
    # funde janelas sobrepostas em episódios
    merged: list[list] = []
    for start, sc, x in flagged:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = start + WINDOW_ML_S - 1
            merged[-1][2] = min(merged[-1][2], sc)
        else:
            merged.append([start, start + WINDOW_ML_S - 1, sc, x])
    for a, b, sc, x in merged:
        # sinal que mais se afasta da população explica o episódio
        ep = Episode("multivariado", a, b, round(float(sc), 3), "atipico", "moderado", "isolation_forest_populacional")
        ep.explanation = _explain(x, pop)  # type: ignore[attr-defined]
        eps.append(ep)
    return eps


def _explain(x: np.ndarray, pop: PopulationModel) -> str:
    means = x[:3]
    return ", ".join(f"{VITAL_LABELS[s]} média {m:.0f} {VITAL_UNITS[s]}" for s, m in zip(ML_SIGNALS, means))


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def _episode_message(ep: Episode) -> str:
    if ep.signal == "multivariado":
        return f"Padrão multivariado atípico vs. população PhysioNet ({getattr(ep, 'explanation', '')})"
    label, unit = VITAL_LABELS[ep.signal], VITAL_UNITS[ep.signal]
    what = {"regra_clinica_news2": "fora da faixa segura", "zscore_robusto_janela_60s": "mudança abrupta",
            "cusum_deriva": "deriva sustentada"}[ep.method]
    arrow = "↓" if ep.direction == "baixo" else "↑"
    extra = f" (basal {ep.reference:.0f})" if hasattr(ep, "reference") else ""
    return f"{label} {arrow} {what}: {ep.extreme:.0f} {unit}{extra} por {ep.duration_s}s"


def _kind(ep: Episode) -> str:
    if ep.signal == "multivariado":
        return "vitais_padrao_atipico"
    return f"{ep.signal.lower()}_{ep.direction}_{ep.method.split('_')[0]}"


def consolidate(episodes: list[Episode], gap_s: int = 30) -> list[Episode]:
    """Funde episódios do mesmo sinal/direção/método separados por menos de ``gap_s``.

    Evita que um quadro contínuo que oscila em torno do limiar (FR 25↔26)
    vire uma rajada de alertas fragmentados — a equipe recebe um episódio só,
    com a pior severidade e o valor extremo do período.
    """
    from src.models import SEVERITY_ORDER

    groups: dict[tuple, list[Episode]] = {}
    for ep in sorted(episodes, key=lambda e: e.start_s):
        method_gap = gap_s * 2 if ep.method.startswith("isolation") else gap_s
        key = (ep.signal, ep.direction, ep.method)
        bucket = groups.setdefault(key, [])
        if bucket and ep.start_s - bucket[-1].end_s <= method_gap:
            last = bucket[-1]
            last.end_s = max(last.end_s, ep.end_s)
            if SEVERITY_ORDER[ep.severity] > SEVERITY_ORDER[last.severity]:
                last.severity = ep.severity
            pick = min if (ep.direction == "baixo" or ep.method.startswith("isolation")) else max
            last.extreme = pick(last.extreme, ep.extreme)
        else:
            bucket.append(ep)
    return sorted((e for b in groups.values() for e in b), key=lambda e: (e.start_s, e.signal))


def detect(df: pd.DataFrame, patient_id: str | None = None, population: PopulationModel | None = None) -> tuple[list[Finding], list[Episode]]:
    episodes = rule_based(df) + robust_zscore(df) + cusum(df)
    if population is not None:
        episodes += isolation_forest(df, population)
    episodes = consolidate(episodes)
    findings = [
        Finding(
            modality="vitals", kind=_kind(ep), severity=ep.severity, patient_id=patient_id,
            message=_episode_message(ep), time_ref=f"t={ep.start_s}s–{ep.end_s}s", method=ep.method,
            evidence={"sinal": ep.signal, "inicio_s": ep.start_s, "fim_s": ep.end_s, "extremo": ep.extreme},
        )
        for ep in episodes
    ]
    return findings, episodes


def summary(df: pd.DataFrame, findings: list[Finding]) -> dict:
    stats = {
        sig: {"min": round(float(df[sig].min()), 1), "media": round(float(df[sig].mean()), 1),
              "max": round(float(df[sig].max()), 1)}
        for sig in SIGNALS if sig in df
    }
    return {
        "fonte": df.attrs.get("source"),
        "duracao_s": int(df["time_s"].iloc[-1]),
        "amostras_faltantes_interpoladas": df.attrs.get("missing", 0),
        "estatisticas": stats,
        "n_achados": len(findings),
        "risco_vitais": modality_risk(findings),
    }
