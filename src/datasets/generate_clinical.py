"""Gera os dados clínicos estruturados (sintéticos e determinísticos).

O PhysioNet BIDMC fornece FC, SpO₂ e FR reais, mas não traz pressão
arterial, prescrições nem dados de movimentação durante a internação —
esses três conjuntos são gerados aqui com semente fixa, cada paciente com
um "roteiro clínico" conhecido, para que as anomalias detectadas possam
ser conferidas contra o que foi de fato injetado (ground truth).

Saídas (em ``data/clinical/``):
    pacientes.csv          — cadastro + mapeamento para os registros BIDMC
    pressao_arterial.csv   — PA sistólica/diastólica a 1 Hz (linha arterial)
    prescricoes.csv        — histórico de prescrições
    movimentacao.csv       — actigrafia de 48h (contagem de atividade a cada 5 min)
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta

import numpy as np

from src.config import BLOOD_PRESSURE_CSV, CLINICAL_DIR, MOVEMENT_CSV, PATIENTS_CSV, PRESCRIPTIONS_CSV

SEED = 42
ADMISSION = datetime(2026, 10, 10, 8, 0, 0)

PATIENTS = [
    {
        "patient_id": "PAC-0001", "idade": 45, "sexo": "M",
        "diagnostico": "Pós-operatório de colecistectomia videolaparoscópica",
        "setor": "Enfermaria cirúrgica", "alergias": "nenhuma",
        "bidmc_record": "02", "audio": "PAC-0001_consulta.wav", "video": "PAC-0001_fisio_ombro.mp4",
        "cenario": "Controle — evolução esperada, sem anomalias injetadas",
    },
    {
        "patient_id": "PAC-0002", "idade": 72, "sexo": "F",
        "diagnostico": "DPOC exacerbada",
        "setor": "UTI", "alergias": "dipirona",
        "bidmc_record": "32", "audio": "PAC-0002_consulta.wav", "video": "",
        "cenario": "Hipoxemia persistente real (BIDMC 32) + voz com sinais respiratórios + salto de corticoide",
    },
    {
        "patient_id": "PAC-0003", "idade": 68, "sexo": "M",
        "diagnostico": "AVC isquêmico em reabilitação",
        "setor": "Enfermaria neurológica", "alergias": "nenhuma",
        "bidmc_record": "45", "audio": "PAC-0003_consulta.wav", "video": "PAC-0003_fisio_ombro.mp4",
        "cenario": "FC basal baixa com episódios de bradicardia (BIDMC 45, 44–72 bpm) + disartria + assimetria na fisioterapia + imobilidade prolongada",
    },
    {
        "patient_id": "PAC-0004", "idade": 59, "sexo": "F",
        "diagnostico": "Sepse de foco urinário",
        "setor": "UTI", "alergias": "penicilina",
        "bidmc_record": "25", "audio": "PAC-0004_consulta.wav", "video": "",
        "cenario": "Taquicardia real (BIDMC 25) + hipotensão + fadiga vocal + prescrição com conflito de alergia",
    },
    {
        "patient_id": "PAC-0005", "idade": 81, "sexo": "F",
        "diagnostico": "Pneumonia comunitária, alto risco de queda",
        "setor": "Enfermaria clínica", "alergias": "sulfa",
        "bidmc_record": "22", "audio": "PAC-0005_consulta.wav", "video": "PAC-0005_fisio_agachamento.mp4",
        "cenario": "Dessaturações reais (BIDMC 22) + fadiga + perda de equilíbrio na fisio + agitação noturna e queda",
    },
    {
        "patient_id": "PAC-0006", "idade": 63, "sexo": "M",
        "diagnostico": "Pós-operatório de revascularização do miocárdio",
        "setor": "UTI cardiológica", "alergias": "nenhuma",
        "bidmc_record": "46", "audio": "PAC-0006_consulta.wav", "video": "PAC-0006_cirurgia.mp4",
        "cenario": "Taquipneia real (BIDMC 46) + falha de contagem de instrumental na cirurgia + suspensão abrupta de heparina",
    },
]


def write_patients() -> None:
    with PATIENTS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(PATIENTS[0].keys()))
        writer.writeheader()
        writer.writerows(PATIENTS)


def write_blood_pressure(rng: np.random.Generator, duration_s: int = 481) -> None:
    """PA contínua (1 Hz), alinhada ao tempo dos registros BIDMC (8 min)."""
    baselines = {
        "PAC-0001": (122, 78), "PAC-0002": (134, 82), "PAC-0003": (148, 88),
        "PAC-0004": (108, 64), "PAC-0005": (128, 74), "PAC-0006": (118, 70),
    }
    t = np.arange(duration_s)
    with BLOOD_PRESSURE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient_id", "time_s", "SBP", "DBP"])
        for pid, (sbp0, dbp0) in baselines.items():
            # Variação respiratória (~0.25 Hz) + ruído do transdutor
            sbp = sbp0 + 3 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 1.5, duration_s)
            dbp = dbp0 + 2 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 1.0, duration_s)
            if pid == "PAC-0004":
                # Choque séptico: queda progressiva da PA a partir de t=200s
                drop = np.clip((t - 200) / 120, 0, 1)
                sbp -= 30 * drop
                dbp -= 20 * drop
            for i in range(duration_s):
                writer.writerow([pid, int(t[i]), round(float(sbp[i]), 1), round(float(dbp[i]), 1)])


def write_prescriptions() -> None:
    def ts(hours: float) -> str:
        return (ADMISSION + timedelta(hours=hours)).isoformat()

    rows = [
        # PAC-0001 — evolução esperada
        ("PAC-0001", ts(0), "dipirona", "analgesico", 1000, 6, "EV", "iniciar", "Dr. Ramos"),
        ("PAC-0001", ts(0), "omeprazol", "protetor_gastrico", 40, 24, "EV", "iniciar", "Dr. Ramos"),
        ("PAC-0001", ts(24), "dipirona", "analgesico", 1000, 6, "VO", "ajustar", "Dr. Ramos"),
        ("PAC-0001", ts(48), "omeprazol", "protetor_gastrico", 20, 24, "VO", "ajustar", "Dr. Ramos"),
        # PAC-0002 — DPOC: salto inesperado de corticoide (40 -> 200 mg)
        ("PAC-0002", ts(0), "metilprednisolona", "corticoide", 40, 12, "EV", "iniciar", "Dra. Lima"),
        ("PAC-0002", ts(0), "salbutamol", "broncodilatador", 5, 4, "INAL", "iniciar", "Dra. Lima"),
        ("PAC-0002", ts(12), "metilprednisolona", "corticoide", 40, 12, "EV", "ajustar", "Dra. Lima"),
        ("PAC-0002", ts(30), "metilprednisolona", "corticoide", 200, 12, "EV", "ajustar", "Dr. Souza"),
        # PAC-0003 — AVC: evolução estável
        ("PAC-0003", ts(0), "acido_acetilsalicilico", "antiagregante", 100, 24, "VO", "iniciar", "Dr. Alves"),
        ("PAC-0003", ts(0), "atorvastatina", "estatina", 40, 24, "VO", "iniciar", "Dr. Alves"),
        ("PAC-0003", ts(36), "atorvastatina", "estatina", 80, 24, "VO", "ajustar", "Dr. Alves"),
        # PAC-0004 — Sepse: alergia a penicilina x piperacilina-tazobactam; vancomicina dose dobrada; noradrenalina
        ("PAC-0004", ts(0), "ceftriaxona", "antibiotico", 2000, 24, "EV", "iniciar", "Dra. Costa"),
        ("PAC-0004", ts(20), "ceftriaxona", "antibiotico", 0, 0, "EV", "suspender", "Dra. Costa"),
        ("PAC-0004", ts(20), "piperacilina_tazobactam", "antibiotico", 4500, 6, "EV", "iniciar", "Dr. Souza"),
        ("PAC-0004", ts(20), "vancomicina", "antibiotico", 1000, 12, "EV", "iniciar", "Dr. Souza"),
        ("PAC-0004", ts(32), "vancomicina", "antibiotico", 2500, 12, "EV", "ajustar", "Dr. Souza"),
        ("PAC-0004", ts(34), "noradrenalina", "vasopressor", 8, 0, "EV", "iniciar", "Dra. Costa"),
        # PAC-0005 — Pneumonia: sem anomalia de prescrição (controle negativo dessa modalidade)
        ("PAC-0005", ts(0), "amoxicilina_clavulanato", "antibiotico", 1000, 8, "EV", "iniciar", "Dr. Ramos"),
        ("PAC-0005", ts(0), "enoxaparina", "anticoagulante", 40, 24, "SC", "iniciar", "Dr. Ramos"),
        ("PAC-0005", ts(48), "amoxicilina_clavulanato", "antibiotico", 875, 12, "VO", "ajustar", "Dr. Ramos"),
        # PAC-0006 — Pós-op cardíaco: heparina suspensa sem substituto + duplicidade de anticoagulante
        ("PAC-0006", ts(0), "heparina", "anticoagulante", 25000, 24, "EV", "iniciar", "Dr. Alves"),
        ("PAC-0006", ts(0), "metoprolol", "betabloqueador", 50, 12, "VO", "iniciar", "Dr. Alves"),
        ("PAC-0006", ts(18), "enoxaparina", "anticoagulante", 80, 12, "SC", "iniciar", "Dra. Lima"),
        ("PAC-0006", ts(26), "heparina", "anticoagulante", 0, 0, "EV", "suspender", "Dra. Lima"),
        ("PAC-0006", ts(26), "enoxaparina", "anticoagulante", 0, 0, "SC", "suspender", "Dra. Lima"),
    ]
    with PRESCRIPTIONS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient_id", "timestamp", "medicamento", "classe", "dose_mg", "intervalo_h", "via", "acao", "prescritor"])
        writer.writerows(rows)


def write_movement(rng: np.random.Generator, hours: int = 48, step_min: int = 5) -> None:
    """Actigrafia (pulseira/sensor de leito): contagem de atividade a cada 5 min."""
    n = hours * 60 // step_min
    with MOVEMENT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient_id", "timestamp", "atividade", "saida_leito", "impacto_g"])
        for p in PATIENTS:
            pid = p["patient_id"]
            activity = np.zeros(n)
            bed_exit = np.zeros(n, dtype=int)
            impact = rng.normal(1.0, 0.05, n).clip(0.8, 1.3)
            for i in range(n):
                hour = (ADMISSION + timedelta(minutes=i * step_min)).hour
                awake = 7 <= hour < 22
                activity[i] = rng.gamma(3.0, 12.0) if awake else rng.gamma(1.2, 4.0)
                # Reposicionamento no leito a cada ~2h (pico curto de atividade)
                if i % 24 == 0:
                    activity[i] += 80
            if pid == "PAC-0003":
                # Hemiparesia + imobilidade: atividade muito baixa e sem reposicionamento por 6h (12h-18h do dia 2)
                activity *= 0.35
                start = (24 + 4) * 60 // step_min
                activity[start:start + 6 * 60 // step_min] = rng.gamma(1.0, 1.0, 6 * 60 // step_min)
            if pid == "PAC-0005":
                # Agitação noturna (01h-04h da 2ª noite) + saídas do leito + queda às 03:40
                night = 41 * 60 // step_min  # 41h após a admissão (08h) = 01h do dia 12/10
                activity[night:night + 36] = rng.gamma(4.0, 25.0, 36)
                bed_exit[night + 6] = bed_exit[night + 20] = 1
                fall = night + 32
                impact[fall] = 3.6
                bed_exit[fall - 1] = 1
                activity[fall + 1:fall + 7] = 0.5  # imóvel após a queda
            for i in range(n):
                ts = (ADMISSION + timedelta(minutes=i * step_min)).isoformat()
                writer.writerow([pid, ts, round(float(activity[i]), 1), int(bed_exit[i]), round(float(impact[i]), 2)])


def main() -> None:
    CLINICAL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    write_patients()
    write_blood_pressure(rng)
    write_prescriptions()
    write_movement(rng)
    print(f"Dados clínicos gerados em {CLINICAL_DIR}")


if __name__ == "__main__":
    main()
