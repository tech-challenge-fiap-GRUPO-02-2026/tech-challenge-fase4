"""Anomalias na evolução de prescrições (alterações inesperadas no tratamento).

Regras de farmacovigilância hospitalar + um componente estatístico:

* **Salto de dose** — ajuste ≥ 2,5× da dose diária anterior do mesmo
  medicamento (ou corte ≤ 0,4× em classe crítica), ou dose diária acima do
  máximo de referência; titulações usuais (2×) e desmames não disparam;
* **Conflito de alergia** — medicamento da mesma família de uma alergia
  registrada (ex.: penicilina × piperacilina);
* **Medicamento de alta vigilância iniciado** (ISMP: vasopressores,
  anticoagulantes, insulina, opioides, KCl) — exige dupla checagem;
* **Duplicidade terapêutica** — dois ativos da mesma classe crítica
  simultâneos (ex.: heparina + enoxaparina);
* **Suspensão abrupta** de classe crítica sem substituto (ex.: retirar
  todo anticoagulante no pós-operatório cardíaco);
* **Z-score populacional** — dose fora de ±3σ (em log) das doses do mesmo
  medicamento no hospital, para pegar o que as regras não cobrem.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PRESCRIPTIONS_CSV
from src.models import Finding, modality_risk

HIGH_ALERT = {"noradrenalina", "heparina", "enoxaparina", "insulina", "morfina", "fentanil", "cloreto_potassio"}
CRITICAL_CLASSES = {"anticoagulante", "vasopressor", "antibiotico"}
DOSE_JUMP_RATIO = 2.5
DOSE_CUT_RATIO = 0.4
# dose diária máxima de referência (mg/dia) — bulas/UpToDate, simplificado para o protótipo
MAX_DAILY_MG = {
    "vancomicina": 4000, "metilprednisolona": 1000, "dipirona": 4000, "piperacilina_tazobactam": 18000,
    "ceftriaxona": 4000, "enoxaparina": 200, "heparina": 40000, "atorvastatina": 80, "omeprazol": 80,
}
ALLERGY_FAMILIES = {
    "penicilina": {"amoxicilina", "amoxicilina_clavulanato", "ampicilina", "piperacilina_tazobactam", "oxacilina"},
    "sulfa": {"sulfametoxazol_trimetoprima", "sulfadiazina"},
    "dipirona": {"dipirona"},
}


def _finding(kind, severity, patient_id, message, time_ref, method, evidence) -> Finding:
    return Finding(modality="prescriptions", kind=kind, severity=severity, message=message, patient_id=patient_id,
                   time_ref=time_ref, method=method, evidence=evidence)


def load(path=PRESCRIPTIONS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)


def daily_dose(row) -> float:
    if row["intervalo_h"] and row["intervalo_h"] > 0:
        return row["dose_mg"] * 24 / row["intervalo_h"]
    return row["dose_mg"]


def _active_at(df: pd.DataFrame, ts) -> dict[str, pd.Series]:
    """Último estado de cada medicamento até ts (suspensos excluídos)."""
    state: dict[str, pd.Series] = {}
    for _, row in df[df["timestamp"] <= ts].iterrows():
        if row["acao"] == "suspender":
            state.pop(row["medicamento"], None)
        else:
            state[row["medicamento"]] = row
    return state


def population_stats(df: pd.DataFrame) -> dict[str, tuple[float, float, int]]:
    stats = {}
    active = df[df["acao"] != "suspender"]
    for drug, grp in active.groupby("medicamento"):
        logs = np.log(grp.apply(daily_dose, axis=1).clip(lower=1e-3))
        stats[drug] = (float(logs.mean()), float(logs.std(ddof=0)), len(grp))
    return stats


def detect(df: pd.DataFrame, patient_id: str, allergies: str = "", pop_stats: dict | None = None) -> list[Finding]:
    pat = df[df["patient_id"] == patient_id].reset_index(drop=True)
    findings: list[Finding] = []
    allergy_set = {a.strip().lower() for a in str(allergies).split(",") if a.strip() and a.strip() != "nenhuma"}
    last_by_drug: dict[str, pd.Series] = {}
    suspension_reported: set[tuple] = set()

    for _, row in pat.iterrows():
        drug, ts = row["medicamento"], row["timestamp"]
        tref = ts.isoformat(timespec="minutes")
        ev = {"medicamento": drug, "dose_mg": row["dose_mg"], "intervalo_h": row["intervalo_h"], "prescritor": row["prescritor"]}

        if row["acao"] == "suspender":
            last_by_drug.pop(drug, None)
            key = (ts, row["classe"])
            if row["classe"] in CRITICAL_CLASSES and key not in suspension_reported:
                remaining = [r for r in _active_at(pat, ts).values() if r["classe"] == row["classe"]]
                if not remaining:
                    suspended = pat[(pat["timestamp"] == ts) & (pat["acao"] == "suspender") & (pat["classe"] == row["classe"])]
                    names = " e ".join(suspended["medicamento"])
                    suspension_reported.add(key)
                    findings.append(_finding(f"suspensao_{row['classe']}_sem_substituto", "alto", patient_id,
                        f"Suspensão de {names} deixa o paciente sem nenhum {row['classe']} ativo", tref,
                        "regra: suspensão abrupta de classe crítica", ev))
            continue

        for allergy in allergy_set:
            if drug in ALLERGY_FAMILIES.get(allergy, set()) or allergy in drug:
                findings.append(_finding("conflito_alergia", "critico", patient_id,
                    f"{drug} prescrito para paciente com alergia registrada a {allergy}", tref,
                    "regra: família farmacológica × alergias do prontuário", ev))

        prev = last_by_drug.get(drug)
        if prev is not None:
            ratio = daily_dose(row) / max(daily_dose(prev), 1e-6)
            # titulação usual (estatina 40→80) e desmame (EV→VO) não são anomalias;
            # salto ≥ 2,5× ou corte ≤ 0,4× de classe crítica são.
            if ratio >= DOSE_JUMP_RATIO or (ratio <= DOSE_CUT_RATIO and row["classe"] in CRITICAL_CLASSES):
                findings.append(_finding("salto_de_dose", "alto", patient_id,
                    f"Dose diária de {drug} mudou {ratio:.1f}× ({daily_dose(prev):.0f} → {daily_dose(row):.0f} mg/dia) "
                    f"por {row['prescritor']}", tref,
                    f"regra: variação ≥ {DOSE_JUMP_RATIO}× entre prescrições consecutivas",
                    {**ev, "razao": round(ratio, 2)}))
        elif row["acao"] == "iniciar" and drug in HIGH_ALERT and row["timestamp"] > pat["timestamp"].min():
            findings.append(_finding("inicio_alta_vigilancia", "moderado", patient_id,
                f"Início de medicamento de alta vigilância ({drug}) — exige dupla checagem", tref,
                "regra: lista ISMP de medicamentos potencialmente perigosos", ev))

        max_daily = MAX_DAILY_MG.get(drug)
        if max_daily and daily_dose(row) > max_daily:
            findings.append(_finding("dose_acima_maxima", "critico", patient_id,
                f"{drug}: {daily_dose(row):.0f} mg/dia acima do máximo de referência ({max_daily} mg/dia)", tref,
                "regra: dose diária máxima", ev))

        if pop_stats and drug in pop_stats:
            mu, sigma, n = pop_stats[drug]
            if n >= 3 and sigma > 0:
                z = (np.log(max(daily_dose(row), 1e-3)) - mu) / sigma
                if abs(z) > 3:
                    findings.append(_finding("dose_atipica_populacional", "moderado", patient_id,
                        f"Dose de {drug} atípica para o hospital (z={z:.1f})", tref, "z-score log-dose populacional",
                        {**ev, "z": round(float(z), 2)}))

        active_same_class = [r for d, r in _active_at(pat, ts).items()
                             if d != drug and r["classe"] == row["classe"] and row["classe"] == "anticoagulante"]
        if active_same_class:
            findings.append(_finding("duplicidade_terapeutica", "alto", patient_id,
                f"{drug} iniciado com {active_same_class[0]['medicamento']} ainda ativo (duplicidade de {row['classe']})",
                tref, "regra: duas drogas ativas da mesma classe crítica", ev))
        last_by_drug[drug] = row
    return findings


def summary(df: pd.DataFrame, patient_id: str, findings: list[Finding]) -> dict:
    pat = df[df["patient_id"] == patient_id]
    final = _active_at(pat, pat["timestamp"].max()) if not pat.empty else {}
    return {
        "n_prescricoes": int(len(pat)),
        "ativos_atuais": sorted(final),
        "n_achados": len(findings),
        "risco_prescricoes": modality_risk(findings),
    }
