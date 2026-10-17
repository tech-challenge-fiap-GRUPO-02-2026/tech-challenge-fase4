"""Avaliação quantitativa dos detectores contra o ground truth conhecido.

1. **Eventos acústicos** (tosse/respiração) — precisão/recall com
   tolerância de 0,6 s contra os eventos injetados (``.json`` de cada áudio);
2. **Indicadores vocais** — perfil injetado × flags disparadas;
3. **Sinais vitais com injeção controlada** — eventos sintéticos
   (dessaturação, taquicardia, bradicardia, taquipneia, hipotensão)
   inseridos num registro PhysioNet estável (BIDMC 02), em lote e em tempo
   real: recall, falsos positivos e latência de disparo;
4. **Achados esperados por paciente** — para vídeo, prescrições,
   movimentação e síndromes da fusão: o tipo esperado foi detectado? houve
   achado inesperado?

    py -m src.main evaluate        → reports/avaliacao.md + reports/output/avaliacao.json
"""
from __future__ import annotations

import json

import numpy as np

from src.config import AUDIO_DIR, OUTPUT_DIR, REPORTS_DIR

EXPECTED = {  # modalidade → paciente → prefixos de kind esperados
    "video": {
        "PAC-0001": [], "PAC-0003": ["adm_reduzida", "assimetria", "compensacao_tronco", "tremor_punho_esquerdo"],
        "PAC-0005": ["instabilidade_postural", "queda_detectada"],
        "PAC-0006": ["falha_contagem", "instrumento_fora", "quebra_area_esteril"],
    },
    "prescriptions": {
        "PAC-0001": [], "PAC-0002": ["salto_de_dose"], "PAC-0003": [],
        "PAC-0004": ["conflito_alergia", "salto_de_dose", "dose_acima_maxima", "inicio_alta_vigilancia"],
        "PAC-0005": [], "PAC-0006": ["duplicidade_terapeutica", "suspensao_anticoagulante", "inicio_alta_vigilancia"],
    },
    "movement": {
        "PAC-0001": [], "PAC-0002": [], "PAC-0003": ["imobilidade_prolongada"], "PAC-0004": [],
        "PAC-0005": ["agitacao_noturna", "queda_leito"], "PAC-0006": [],
    },
    "fusion": {
        "PAC-0001": [], "PAC-0002": ["insuficiencia_respiratoria"], "PAC-0003": ["suspeita_avc"],
        "PAC-0004": ["possivel_choque", "risco_seguranca_medicamentosa"],
        "PAC-0005": ["queda_confirmada", "insuficiencia_respiratoria", "delirium"],
        "PAC-0006": ["risco_seguranca_medicamentosa"],
    },
}
EXPECTED_VOICE = {
    "normal": set(), "respiratorio": {"voz_respiratorio"}, "disartria": {"voz_disartria"},
    "fadiga": {"voz_fadiga"}, "fadiga_respiratorio": {"voz_fadiga", "voz_respiratorio"},
}

# eventos injetados no registro estável: (sinal, início_s, duração_s, valor-alvo)
INJECTIONS = [
    ("SpO2", 60, 40, 85.0),
    ("HR", 150, 30, 142.0),
    ("RESP", 230, 35, 30.0),
    ("HR", 320, 25, 38.0),
    ("SBP", 400, 40, 78.0),
]


def _match_events(pred: list[tuple[str, float]], truth: list[tuple[str, float]], tol: float = 0.6) -> dict:
    used = set()
    tp = 0
    for label, t in truth:
        for j, (pl, pt) in enumerate(pred):
            if j not in used and pl == label and abs(pt - t) <= tol:
                used.add(j)
                tp += 1
                break
    fp, fn = len(pred) - tp, len(truth) - tp
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precisao": round(prec, 3), "recall": round(rec, 3)}


def evaluate_audio(use_audioset_model: bool = False) -> dict:
    from src.audio.features import extract_features
    from src.audio.voice_analyzer import assess, cough_episodes
    from src.audio.wav_io import read_wav

    per_label = {"Cough": [], "Breathing": []}
    voice_rows, all_pred, all_truth = [], [], []
    for meta_path in sorted(AUDIO_DIR.glob("*_consulta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feats = extract_features(*read_wav(meta_path.with_suffix(".wav")))
        # tosse: compara episódios (rajadas agrupadas) com episódios injetados
        coughs, last_end = [], float("-inf")
        for e in sorted((e for e in feats.events if e.label == "Cough"), key=lambda e: e.start_s):
            if e.start_s - last_end > 0.6:
                coughs.append(("Cough", e.start_s))
            last_end = e.end_s
        pred = coughs + [("Breathing", e.start_s) for e in feats.events if e.label in ("Breathing", "Sigh")]
        truth = [(e["tipo"], e["inicio_s"]) for e in meta["eventos_injetados"]]
        all_pred += [(l, t, meta["patient_id"]) for l, t in pred]
        all_truth += [(l, t, meta["patient_id"]) for l, t in truth]
        for label in per_label:
            per_label[label].append(_match_events([p for p in pred if p[0] == label], [t for t in truth if t[0] == label]))
        flags = set(assess(feats).flags)
        expected = EXPECTED_VOICE[meta["perfil_injetado"]]
        voice_rows.append({"patient_id": meta["patient_id"], "perfil": meta["perfil_injetado"],
                           "esperado": sorted(expected), "detectado": sorted(flags),
                           "acerto": expected <= flags, "extras": sorted(flags - expected),
                           "episodios_tosse": cough_episodes(feats)})
    events = {}
    for label, rows in per_label.items():
        tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
        events[label] = {"tp": tp, "fp": fp, "fn": fn,
                         "precisao": round(tp / (tp + fp), 3) if tp + fp else 1.0,
                         "recall": round(tp / (tp + fn), 3) if tp + fn else 1.0}
    out = {"eventos_acusticos": events, "indicadores_vocais": voice_rows,
           "acuracia_indicadores": round(np.mean([r["acerto"] for r in voice_rows]), 3)}

    if use_audioset_model:
        from src.audio import audioset_classifier

        if audioset_classifier.available():
            rows = []
            for meta_path in sorted(AUDIO_DIR.glob("*_consulta.json")):
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                sig, sr = read_wav(meta_path.with_suffix(".wav"))
                summary = audioset_classifier.summarize(audioset_classifier.classify(sig, sr))
                has_resp = "respiratorio" in meta["perfil_injetado"]
                rows.append({"patient_id": meta["patient_id"], "perfil": meta["perfil_injetado"],
                             "classes": {k: v["score_max"] for k, v in summary.items()},
                             "detectou_respiratorio": bool(summary), "esperado": has_resp})
            out["audioset_ast"] = rows
    return out


def evaluate_vitals_injection() -> dict:
    from src.anomaly import vitals
    from src.datasets import physionet
    from src.monitoring.realtime import replay

    df = physionet.load_patient_vitals("PAC-0001", "02").copy()
    for sig, start, dur, target in INJECTIONS:
        idx = (df["time_s"] >= start) & (df["time_s"] < start + dur)
        ramp = np.clip((df.loc[idx, "time_s"] - start) / 5.0, 0, 1)  # rampa de 5 s
        df.loc[idx, sig] = df.loc[idx, sig] + (target - df.loc[idx, sig]) * ramp
    pop =vitals.train_population_model(physionet.load_population(exclude={"02", "32", "45", "25", "22", "46"}))
    batch, _ = vitals.detect(df, "INJ", pop)
    stream, mon = replay(df, "INJ")
    stream = [f for f in stream if f.severity != "info"]

    rows = []
    for sig, start, dur, target in INJECTIONS:
        def hit(f):
            e = f.evidence
            s = e.get("sinal")
            t0 = e.get("inicio_s")
            if t0 is None and f.time_ref:
                t0 = float(f.time_ref.split("=")[1].split("s")[0])
            return s == sig and t0 is not None and start - 5 <= t0 <= start + dur
        b = [f for f in batch if hit(f)]
        s = [f for f in stream if hit(f)]
        delay = None
        if s:
            fired = min(float(f.time_ref.split("=")[1].rstrip("s")) for f in s)
            delay = round(fired - start, 1)
        rows.append({"sinal": sig, "inicio_s": start, "duracao_s": dur, "alvo": target,
                     "lote_detectou": bool(b), "lote_metodos": sorted({f.method for f in b}),
                     "tempo_real_detectou": bool(s), "tempo_real_latencia_s": delay})
    windows = [(s, st - 5, st + d + 30) for s, st, d, _ in INJECTIONS]

    def is_fp(f):
        sig = f.evidence.get("sinal")
        t0 = f.evidence.get("inicio_s")
        if t0 is None:
            return False
        return not any((sig == s or sig == "multivariado") and a <= t0 <= b for s, a, b in windows)
    return {
        "registro_base": "PhysioNet BIDMC 02 (estável)", "injecoes": rows,
        "recall_lote": round(np.mean([r["lote_detectou"] for r in rows]), 3),
        "recall_tempo_real": round(np.mean([r["tempo_real_detectou"] for r in rows]), 3),
        "latencia_media_tempo_real_s": round(float(np.mean([r["tempo_real_latencia_s"] for r in rows if r["tempo_real_latencia_s"] is not None])), 1),
        "falsos_positivos_lote": sum(is_fp(f) for f in batch),
        "falsos_positivos_tempo_real": sum(is_fp(f) for f in stream),
    }


def evaluate_expected() -> dict:
    from src.alerts.alert_manager import AlertManager
    from src.pipeline import MonitoringSystem

    system = MonitoringSystem(alert_manager=AlertManager(persist=False), prefer_azure=False)
    results = {}
    for modality, per_patient in EXPECTED.items():
        rows = []
        for pid, expected in per_patient.items():
            if modality == "fusion":
                findings = system.assess(pid, emit_alerts=False).findings["fusion"]
            else:
                _, findings = getattr(system, f"analyze_{modality}")(pid)
            kinds = [f.kind for f in findings if f.severity != "info"]
            found = [e for e in expected if any(k.startswith(e) for k in kinds)]
            unexpected = [k for k in kinds if not any(k.startswith(e) for e in expected)]
            rows.append({"patient_id": pid, "esperado": expected, "detectado": kinds,
                         "recall": round(len(found) / len(expected), 3) if expected else 1.0,
                         "inesperados": unexpected})
        n_exp = sum(len(r["esperado"]) for r in rows)
        n_found = sum(round(r["recall"] * len(r["esperado"])) for r in rows)
        results[modality] = {"pacientes": rows, "recall": round(n_found / n_exp, 3) if n_exp else 1.0,
                             "achados_inesperados": sum(len(r["inesperados"]) for r in rows)}
    return results


def write_markdown(res: dict) -> str:
    a, v, e = res["audio"], res["vitais_injecao"], res["esperados"]
    L = ["# Avaliação dos detectores", "", "Gerado por `python -m src.main evaluate` — números reais desta execução.", "",
         "## 1. Eventos acústicos (classes AudioSet) vs. eventos injetados", "",
         "| Classe | TP | FP | FN | Precisão | Recall |", "|---|--:|--:|--:|--:|--:|"]
    for label, m in a["eventos_acusticos"].items():
        L.append(f"| {label} | {m['tp']} | {m['fp']} | {m['fn']} | {m['precisao']:.2f} | {m['recall']:.2f} |")
    L += ["", f"## 2. Indicadores vocais — acurácia {a['acuracia_indicadores']:.0%}", "",
          "| Paciente | Perfil injetado | Esperado | Detectado | Acerto |", "|---|---|---|---|:-:|"]
    for r in a["indicadores_vocais"]:
        L.append(f"| {r['patient_id']} | {r['perfil']} | {', '.join(r['esperado']) or '—'} | "
                 f"{', '.join(r['detectado']) or '—'} | {'✅' if r['acerto'] else '❌'} |")
    if "audioset_ast" in a:
        L += ["", "### Classificador AST (AudioSet) — sons respiratórios", "",
              "| Paciente | Perfil | Esperado respiratório | Classes detectadas (score máx) |", "|---|---|:-:|---|"]
        for r in a["audioset_ast"]:
            cls = ", ".join(f"{k} {s:.2f}" for k, s in r["classes"].items()) or "—"
            L.append(f"| {r['patient_id']} | {r['perfil']} | {'sim' if r['esperado'] else 'não'} | {cls} |")
    L += ["", f"## 3. Sinais vitais — injeção controlada em {v['registro_base']}", "",
          f"Recall em lote **{v['recall_lote']:.0%}**, em tempo real **{v['recall_tempo_real']:.0%}**, "
          f"latência média de disparo em tempo real **{v['latencia_media_tempo_real_s']} s** de sinal; "
          f"falsos positivos fora das janelas injetadas: lote {v['falsos_positivos_lote']}, tempo real {v['falsos_positivos_tempo_real']}.", "",
          "| Sinal | Início | Alvo | Lote | Métodos | Tempo real | Latência |", "|---|--:|--:|:-:|---|:-:|--:|"]
    for r in v["injecoes"]:
        L.append(f"| {r['sinal']} | {r['inicio_s']}s | {r['alvo']} | {'✅' if r['lote_detectou'] else '❌'} | "
                 f"{', '.join(r['lote_metodos'])} | {'✅' if r['tempo_real_detectou'] else '❌'} | {r['tempo_real_latencia_s']}s |")
    L += ["", "## 4. Achados esperados por paciente", "", "| Modalidade | Recall | Achados inesperados |", "|---|--:|--:|"]
    for m, r in e.items():
        L.append(f"| {m} | {r['recall']:.0%} | {r['achados_inesperados']} |")
    for m, r in e.items():
        L += ["", f"### {m}", "", "| Paciente | Esperado | Detectado |", "|---|---|---|"]
        for row in r["pacientes"]:
            L.append(f"| {row['patient_id']} | {', '.join(row['esperado']) or '—'} | {', '.join(row['detectado']) or '—'} |")
    return "\n".join(L) + "\n"


def main(use_audioset_model: bool = False) -> dict:
    res = {"audio": evaluate_audio(use_audioset_model), "vitais_injecao": evaluate_vitals_injection(),
           "esperados": evaluate_expected()}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "avaliacao.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md = write_markdown(res)
    (REPORTS_DIR / "avaliacao.md").write_text(md, encoding="utf-8")
    print(md)
    return res


if __name__ == "__main__":
    main()
