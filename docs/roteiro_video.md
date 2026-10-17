# Roteiro do vídeo de demonstração (≤ 15 min)

Requisitos do enunciado cobertos: análise prática de áudio e vídeo · detecção e
resposta a anomalias · integração dos serviços Azure · fluxo final do alerta à equipe.

| Tempo | Bloco | O que mostrar | Comando / tela |
|---|---|---|---|
| 0:00–1:00 | Abertura | Equipe, desafio da Fase 4, visão do fluxo multimodal | `index.html` + diagrama de `docs/architecture.md` |
| 1:00–2:30 | Datasets | PhysioNet BIDMC (53 registros reais), AudioSet (ontologia/catálogo clínico), ESC-50 | `py -m src.main download-datasets` · `data/raw/audioset/clinical_sound_catalog.json` |
| 2:30–5:00 | **Vídeo** | Fisioterapia normal × hemiparesia (PAC-0003): ADM 75° × 158°, compensação de tronco, tremor; queda no agachamento (PAC-0005); cirurgia com tesoura retida (PAC-0006). YOLOv8 real em imagem pública | `py -m src.main video --patient-id PAC-0003` · `reports/output/video/*.md` · `python scripts/validate_yolo.py` |
| 5:00–7:30 | **Áudio + Azure** | Ouvir PAC-0003 (disartria) e PAC-0002 (falta de ar + tosse real); biomarcadores; AST/AudioSet; Speech to Text e Text Analytics (ou fallback, explicando o `provider`) | `py -m src.main azure-check` · `py -m src.main audio --patient-id PAC-0002 --audioset-model` |
| 7:30–9:30 | **Anomalias** | Vitais reais (BIDMC 32: SpO₂ 83%), prescrição com alergia a penicilina (PAC-0004), queda noturna na actigrafia (PAC-0005); avaliação com injeção controlada | `py -m src.main vitals --patient-id PAC-0004` · `py -m src.main evaluate` |
| 9:30–12:30 | **Tempo real + alerta** | Painel: iniciar monitor do PAC-0004, alertas P1/P2 surgindo, reconhecer alerta; síndrome "Possível choque" na fusão | `iniciar_painel.bat` |
| 12:30–14:00 | **Fluxo do alerta** | Roteamento P1→time de resposta rápida, webhook Teams/Logic Apps, trilha de auditoria | `py -m src.main alerts` · `py -m src.main logs` |
| 14:00–15:00 | Fechamento | Resultados (`reports/avaliacao.md`), limitações, próximos passos | `reports/relatorio_tecnico.md` |
