"""Monitoramento de sinais vitais em tempo real (streaming, amostra a amostra).

A detecção em lote (``anomaly/vitals.py``) analisa a série inteira; aqui o
monitor só conhece o passado — exatamente como à beira do leito. A cada
amostra (1 Hz):

* **Regra clínica com persistência** — o alerta só dispara depois de
  ``VITAL_PERSISTENCE_S`` segundos consecutivos fora da faixa (filtra
  artefato de sensor) e é **rearmado** quando o sinal volta ao normal;
* **Z-score online (EWMA)** — média e variância exponenciais por sinal;
  desvio > 4σ sustentado por 5 s sinaliza mudança abrupta;
* **Latência** medida em segundos de sinal entre o início do desvio e o
  disparo do alerta (reportada no relatório técnico).

``replay`` reproduz um registro PhysioNet como se fosse o monitor
multiparamétrico ao vivo, opcionalmente acelerado.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.config import VITAL_LABELS, VITAL_PERSISTENCE_S, VITAL_THRESHOLDS, VITAL_UNITS
from src.models import Finding

SIGNALS = ["HR", "SpO2", "RESP", "SBP", "DBP"]


@dataclass
class _SignalState:
    mean: float | None = None
    var: float = 0.0
    n: int = 0
    rule_run: int = 0
    rule_level: str | None = None   # nível do episódio de regra em curso
    rule_fired: bool = False
    z_run: int = 0
    z_fired: bool = False
    run_start: float | None = None


@dataclass
class OnlineVitalsMonitor:
    patient_id: str
    persistence_s: int = VITAL_PERSISTENCE_S
    alpha: float = 0.05            # EWMA (~20 s de memória)
    z_threshold: float = 4.0
    z_persistence_s: int = 5
    warmup_s: int = 30
    states: dict[str, _SignalState] = field(default_factory=lambda: {s: _SignalState() for s in SIGNALS})
    latencies: list[float] = field(default_factory=list)

    def _classify(self, sig: str, v: float) -> tuple[str | None, str]:
        lo_c, lo_w, hi_w, hi_c = VITAL_THRESHOLDS[sig]
        if v < lo_c:
            return "critico", "baixo"
        if v < lo_w:
            return "alto", "baixo"
        if v > hi_c:
            return "critico", "alto"
        if v > hi_w:
            return "alto", "alto"
        return None, ""

    def update(self, t: float, sample: dict[str, float]) -> list[Finding]:
        out: list[Finding] = []
        for sig in SIGNALS:
            v = sample.get(sig)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            st = self.states[sig]
            label, unit = VITAL_LABELS[sig], VITAL_UNITS[sig]

            # --- regra clínica com persistência e rearme
            level, direction = self._classify(sig, v)
            if level:
                if st.rule_run == 0:
                    st.run_start = t
                st.rule_run += 1
                escalated = st.rule_fired and level == "critico" and st.rule_level != "critico"
                if (st.rule_run >= self.persistence_s and not st.rule_fired) or escalated:
                    st.rule_fired, st.rule_level = True, level
                    self.latencies.append(t - (st.run_start or t))
                    arrow = "↓" if direction == "baixo" else "↑"
                    out.append(Finding(
                        "vitals", f"{sig.lower()}_{direction}_tempo_real", level, patient_id=self.patient_id,
                        message=f"{label} {arrow} {v:.0f} {unit} há {st.rule_run}s (tempo real)",
                        time_ref=f"t={t:.0f}s", method=f"regra clínica + persistência {self.persistence_s}s (streaming)",
                        evidence={"sinal": sig, "valor": v, "inicio_s": st.run_start, "latencia_s": t - (st.run_start or t)},
                    ))
            else:
                if st.rule_fired:
                    out.append(Finding(
                        "vitals", f"{sig.lower()}_normalizado", "info", patient_id=self.patient_id,
                        message=f"{label} voltou à faixa segura ({v:.0f} {unit})", time_ref=f"t={t:.0f}s",
                        method="rearme do alarme",
                    ))
                st.rule_run, st.rule_fired, st.rule_level = 0, False, None

            # --- z-score online (EWMA)
            if st.mean is None:
                st.mean, st.var = v, 1.0
            std = max(np.sqrt(st.var), {"HR": 1.5, "SpO2": 0.5, "RESP": 1.0, "SBP": 2.0, "DBP": 1.5}[sig])
            z = (v - st.mean) / std
            st.n += 1
            if st.n > self.warmup_s and abs(z) > self.z_threshold:
                st.z_run += 1
                if st.z_run >= self.z_persistence_s and not st.z_fired:
                    st.z_fired = True
                    out.append(Finding(
                        "vitals", f"{sig.lower()}_mudanca_abrupta", "moderado", patient_id=self.patient_id,
                        message=f"{label}: mudança abrupta para {v:.0f} {unit} (média recente {st.mean:.0f}, z={z:.1f})",
                        time_ref=f"t={t:.0f}s", method="z-score EWMA online", evidence={"sinal": sig, "z": round(z, 2)},
                    ))
            else:
                st.z_run = 0
                if abs(z) < 2:
                    st.z_fired = False
            # atualiza a média só com amostras não-anômalas (evita "aprender" o evento)
            if abs(z) <= self.z_threshold or st.n <= self.warmup_s:
                diff = v - st.mean
                st.mean += self.alpha * diff
                st.var = (1 - self.alpha) * (st.var + self.alpha * diff * diff)
        return out


def iter_samples(df: pd.DataFrame) -> Iterable[tuple[float, dict]]:
    cols = [c for c in SIGNALS if c in df]
    for row in df[["time_s", *cols]].itertuples(index=False):
        yield float(row[0]), {c: float(v) for c, v in zip(cols, row[1:])}


def replay(df: pd.DataFrame, patient_id: str, on_sample: Callable[[float, dict], None] | None = None,
           on_finding: Callable[[Finding], None] | None = None, speed: float = 0.0,
           should_stop: Callable[[], bool] | None = None) -> tuple[list[Finding], OnlineVitalsMonitor]:
    """Reproduz o registro como stream. speed=0 → o mais rápido possível; speed=10 → 10× tempo real."""
    monitor = OnlineVitalsMonitor(patient_id)
    findings: list[Finding] = []
    for t, sample in iter_samples(df):
        if should_stop and should_stop():
            break
        if on_sample:
            on_sample(t, sample)
        for f in monitor.update(t, sample):
            findings.append(f)
            if on_finding:
                on_finding(f)
        if speed > 0:
            time.sleep(1.0 / speed)
    return findings, monitor
