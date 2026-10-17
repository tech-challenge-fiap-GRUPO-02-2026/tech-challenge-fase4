"""CLI do Monitoramento Multimodal — Tech Challenge Fase 4.

Dados:
    py -m src.main download-datasets          # PhysioNet BIDMC + AudioSet + ESC-50
    py -m src.main generate-data              # dados clínicos, consultas (áudio) e vídeos
Análises por modalidade:
    py -m src.main video   --patient-id PAC-0003 | --file meu_video.mp4 --exercise abducao_ombro
    py -m src.main audio   --patient-id PAC-0003 [--audioset-model]
    py -m src.main vitals  --patient-id PAC-0004
    py -m src.main prescriptions --patient-id PAC-0004
    py -m src.main movement --patient-id PAC-0005
Fluxo completo:
    py -m src.main assess  --patient-id PAC-0005  # fusão + alertas
    py -m src.main run-all                        # todos os pacientes → reports/
    py -m src.main monitor --patient-id PAC-0004 --speed 20   # tempo real no terminal
    py -m src.main serve                          # painel web da equipe médica
Infra:
    py -m src.main azure-check | alerts [--n 20] | logs [--n 20] | evaluate
"""
from __future__ import annotations

import argparse
import json
import sys


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _findings(findings) -> None:
    from src.models import SEVERITY_ORDER

    if not findings:
        print("  (nenhuma anomalia)")
    for f in sorted(findings, key=lambda f: -SEVERITY_ORDER[f.severity]):
        print(f"  [{f.severity.upper():8}] {f.message}" + (f"  ({f.time_ref})" if f.time_ref else ""))
        if f.method:
            print(f"             ↳ {f.method}")


def cmd_download(args) -> None:
    from src.datasets import audioset, esc50, physionet

    print(f"PhysioNet BIDMC: {physionet.download(args.n)} registro(s) novo(s) baixado(s)")
    audioset.download()
    cat = audioset.build_catalog()
    print(f"AudioSet: ontologia com {cat['total_classes_ontologia']} classes, {len(cat['classes_clinicas'])} clínicas catalogadas")
    print(f"ESC-50: {len(esc50.download())} clipes reais de tosse/respiração")


def cmd_generate(args) -> None:
    from src.datasets import generate_audio, generate_clinical, generate_video

    generate_clinical.main()
    if not args.skip_audio:
        generate_audio.main()
    if not args.skip_video:
        generate_video.main()


def _system(args):
    from src.alerts.alert_manager import AlertManager
    from src.pipeline import MonitoringSystem

    return MonitoringSystem(alert_manager=AlertManager(console=getattr(args, "console", False)),
                            prefer_azure=not getattr(args, "no_azure", False),
                            use_audioset_model=getattr(args, "audioset_model", False))


def cmd_video(args) -> None:
    from src.video.video_pipeline import analyze_video

    if args.file:
        report = analyze_video(args.file, args.patient_id, exercise=args.exercise, prefer=args.source,
                               stride=args.stride, weights=args.weights)
    else:
        from src.config import VIDEO_DIR
        from src.pipeline import load_patients

        p = load_patients().set_index("patient_id").loc[args.patient_id]
        if not p["video"]:
            sys.exit(f"{args.patient_id} não tem vídeo associado")
        report = analyze_video(VIDEO_DIR / p["video"], args.patient_id)
    print(f"Vídeo {report.video} · {report.kind} · fonte: {report.source}")
    _findings(report.findings)
    print(f"Relatório: reports/{report.report_path}")
    for img in report.key_frame_images:
        print(f"Frame-chave: reports/{img}")


def cmd_audio(args) -> None:
    from src.audio.audio_pipeline import analyze_consultation
    from src.config import AUDIO_DIR

    path = args.file or AUDIO_DIR / f"{args.patient_id}_consulta.wav"
    r = analyze_consultation(path, args.patient_id, prefer_azure=not args.no_azure, use_audioset_model=args.audioset_model)
    d = r.to_dict()
    print(f"Transcrição ({d['transcricao']['provider']}): {d['transcricao']['texto']}")
    if d["transcricao"]["wer_vs_referencia"] is not None:
        print(f"WER vs. referência: {d['transcricao']['wer_vs_referencia']:.1%}")
    print(f"Texto ({d['analise_texto']['provider']}): sentimento={d['analise_texto']['sentimento']['sentimento']} "
          f"termos críticos={d['analise_texto']['termos_criticos']}")
    print("Indicadores vocais:", d["avaliacao_vocal"]["scores"])
    print("Eventos AudioSet:", [(e["label"], e["start_s"]) for e in d["eventos_audioset"]])
    if d["classificador_audioset_ast"]:
        print("AST (AudioSet):", {k: v["score_max"] for k, v in d["classificador_audioset_ast"]["resumo"].items()})
    _findings(r.findings)
    if args.json:
        _print(d)


def _modality(args, name: str) -> None:
    s = _system(args)
    summary, findings = getattr(s, f"analyze_{name}")(args.patient_id)
    _print(summary)
    _findings(findings)


def cmd_assess(args) -> None:
    s = _system(args)
    a = s.assess(args.patient_id)
    f = a.fusion
    print(f"\n{args.patient_id} — {a.patient['diagnostico']}\nRisco fundido: {f['score']:.2f} ({f['nivel']})")
    print("Por modalidade:", f["risco_por_modalidade"])
    for syn in f["sindromes"]:
        print(f"  🧩 {syn['message']}\n     → {syn['evidence']['acao_sugerida']}")
    print(f"{len(a.alerts)} alerta(s) emitido(s).")
    if args.json:
        _print(a.to_dict())


def cmd_run_all(args) -> None:
    s = _system(args)
    results = s.run_all()
    print(f"{'Paciente':10} {'Risco':>6}  Nível     Alertas  Síndromes")
    for a in results:
        syn = ", ".join(x["kind"] for x in a.fusion["sindromes"]) or "—"
        print(f"{a.patient['patient_id']:10} {a.fusion['score']:6.2f}  {a.fusion['nivel']:9} {len(a.alerts):7}  {syn}")
    print("\nResultados: reports/resultados_execucao.md · JSON por paciente: reports/output/")


def cmd_monitor(args) -> None:
    from src.alerts.alert_manager import AlertManager
    from src.monitoring.realtime import replay
    from src.pipeline import MonitoringSystem

    alerts = AlertManager(console=True)
    s = MonitoringSystem(alert_manager=alerts)
    df = s.vitals_frame(args.patient_id)
    print(f"Monitorando {args.patient_id} — {df.attrs.get('source')} ({len(df)} s) a {args.speed or 'máx'}×\n")

    def on_sample(t, x):
        if int(t) % 30 == 0:
            vals = "  ".join(f"{k}={v:.0f}" for k, v in x.items())
            print(f"\033[90mt={t:4.0f}s  {vals}\033[0m")

    def on_finding(f):
        if f.severity == "info":
            print(f"  ✔ {f.message}")
        else:
            alerts.emit(f)

    findings, mon = replay(df, args.patient_id, on_sample=on_sample, on_finding=on_finding, speed=args.speed)
    lat = mon.latencies
    print(f"\n{len([f for f in findings if f.severity != 'info'])} evento(s); latência média de disparo "
          f"{(sum(lat) / len(lat)) if lat else 0:.1f}s de sinal")


def cmd_serve(args) -> None:
    sys.argv = ["server", "--port", str(args.port)] + (["--open"] if args.open else []) + \
               (["--audioset-model"] if args.audioset_model else [])
    from src.webapp.server import main

    main()


def cmd_azure_check(args) -> None:
    from src.audio.azure_speech import AzureSpeechToText
    from src.audio.azure_text_analytics import AzureTextAnalytics
    from src.cloud.azure_client import AzureError, AzureSettings
    from src.config import AUDIO_DIR

    st = AzureSettings.from_env()
    _print(st.status())
    if st.speech_available:
        try:
            t = AzureSpeechToText(st).transcribe(AUDIO_DIR / "PAC-0001_consulta.wav")
            print("Speech to Text OK:", t.text[:120])
        except AzureError as exc:
            print("Speech to Text FALHOU:", exc)
    if st.language_available:
        try:
            print("Text Analytics OK:", AzureTextAnalytics(st).sentiment("Estou com muita falta de ar e dor no peito."))
        except AzureError as exc:
            print("Text Analytics FALHOU:", exc)
    if not (st.speech_available or st.language_available):
        print("Sem credenciais Azure — o sistema usa os fallbacks locais. Veja .env.example.")


def cmd_alerts(args) -> None:
    from src.alerts.alert_manager import ALERTS_PATH

    if not ALERTS_PATH.exists():
        print("Nenhum alerta registrado.")
        return
    for line in ALERTS_PATH.read_text(encoding="utf-8").splitlines()[-args.n:]:
        a = json.loads(line)
        print(f"{a['created_at'][:19]}  {a['priority']}  {a['patient_id']}  {a['message']}")


def cmd_logs(args) -> None:
    from src.safety.audit_logger import AuditLogger

    for r in AuditLogger().read_recent(args.n):
        print(json.dumps(r, ensure_ascii=False, default=str))


def cmd_evaluate(args) -> None:
    from src.evaluation import main as evaluate_main

    evaluate_main(use_audioset_model=args.audioset_model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitoramento Multimodal — Tech Challenge Fase 4")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download-datasets", help="Baixa PhysioNet BIDMC, AudioSet (ontologia) e ESC-50")
    p.add_argument("--n", type=int, default=53, help="nº de registros BIDMC")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("generate-data", help="Gera dados clínicos, áudios de consulta e vídeos")
    p.add_argument("--skip-audio", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("video", help="Analisa vídeo (fisioterapia/cirurgia)")
    p.add_argument("--patient-id", default=None)
    p.add_argument("--file", default=None, help="vídeo próprio (usa YOLOv8-pose/YOLOv8)")
    p.add_argument("--exercise", choices=["abducao_ombro", "agachamento", "cirurgia"], default=None)
    p.add_argument("--source", choices=["auto", "sidecar", "yolo"], default="auto")
    p.add_argument("--stride", type=int, default=1, help="processa 1 a cada N frames")
    p.add_argument("--weights", default=None, help="pesos YOLOv8 customizados (instrumental cirúrgico)")
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("audio", help="Analisa áudio da consulta (voz + Azure STT + Text Analytics)")
    p.add_argument("--patient-id", default="PAC-0001")
    p.add_argument("--file", default=None)
    p.add_argument("--no-azure", action="store_true")
    p.add_argument("--audioset-model", action="store_true", help="classificador AST treinado no AudioSet")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_audio)

    for name in ("vitals", "prescriptions", "movement"):
        p = sub.add_parser(name, help=f"Detecção de anomalias: {name}")
        p.add_argument("--patient-id", required=True)
        p.set_defaults(func=lambda a, n=name: _modality(a, n))

    p = sub.add_parser("assess", help="Fluxo multimodal completo de um paciente (fusão + alertas)")
    p.add_argument("--patient-id", required=True)
    p.add_argument("--no-azure", action="store_true")
    p.add_argument("--audioset-model", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_assess, console=True)

    p = sub.add_parser("run-all", help="Avalia todos os pacientes e gera os relatórios")
    p.add_argument("--no-azure", action="store_true")
    p.add_argument("--audioset-model", action="store_true")
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser("monitor", help="Monitor de sinais vitais em tempo real (replay PhysioNet)")
    p.add_argument("--patient-id", required=True)
    p.add_argument("--speed", type=float, default=20.0, help="0 = máximo")
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("serve", help="Painel web da equipe médica")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--open", action="store_true")
    p.add_argument("--audioset-model", action="store_true")
    p.set_defaults(func=cmd_serve)

    sub.add_parser("azure-check", help="Verifica credenciais e conectividade Azure").set_defaults(func=cmd_azure_check)
    p = sub.add_parser("alerts", help="Últimos alertas emitidos")
    p.add_argument("--n", type=int, default=20)
    p.set_defaults(func=cmd_alerts)
    p = sub.add_parser("logs", help="Trilha de auditoria")
    p.add_argument("--n", type=int, default=20)
    p.set_defaults(func=cmd_logs)
    p = sub.add_parser("evaluate", help="Avalia detectores contra o ground truth injetado")
    p.add_argument("--audioset-model", action="store_true")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
