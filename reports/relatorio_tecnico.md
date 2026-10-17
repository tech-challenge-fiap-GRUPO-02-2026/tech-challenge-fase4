# Relatório Técnico — Tech Challenge Fase 4
## Monitoramento Multimodal de Pacientes com IA e Azure Cognitive Services

**Pós-Graduação IA Para Desenvolvedores · FIAP · 2026**
**Equipe:** Wilson Lima da Silva · Gustavo Lopes da Silva · Felipe Soeiro Lopes · Vinicius Tavares Sousa da Silva

---

## 1. Objetivo

Com a IA já apoiando exames, documentos e decisões clínicas (Fases 2 e 3), o hospital
quer monitorar continuamente os pacientes a partir de **dados multimodais** — vídeo,
áudio e texto, além de sinais vitais, prescrições e movimentação — para identificar
**sinais precoces de risco** e **alertar a equipe médica em tempo real**.

Entregas técnicas atendidas:

| Requisito do enunciado | Onde foi atendido |
|---|---|
| Processar vídeos clínicos (fisioterapia / cirurgia) | `src/video/` — 3 sessões de fisioterapia + 1 cirurgia |
| OpenPose para análise postural | YOLOv8-pose com conversão para o layout OpenPose COCO-18 (`src/video/pose.py`) |
| YOLOv8 para objetos e áreas críticas | `src/video/objects.py` + `procedure_analysis.py` |
| Relatórios automáticos de desvios/falhas | `reports/output/video/*.md` + frames anotados |
| Processar áudios de consultas e detectar alterações vocais | `src/audio/features.py`, `voice_analyzer.py` |
| Azure Speech to Text | `src/audio/azure_speech.py` |
| Termos críticos e sentimentos com Azure Text Analytics | `src/audio/azure_text_analytics.py` |
| Anomalias em séries de sinais vitais | `src/anomaly/vitals.py` + `src/monitoring/realtime.py` |
| Anomalias na evolução de prescrições | `src/anomaly/prescriptions.py` |
| Anomalias em padrões de movimentação | `src/anomaly/movement.py` (actigrafia) + vídeo |
| Alertas automáticos para a equipe | `src/alerts/alert_manager.py` + painel web |
| Datasets sugeridos (PhysioNet, AudioSet) | `src/datasets/physionet.py`, `audioset.py` |

---

## 2. Descrição do fluxo multimodal

```
 vídeo ──► YOLOv8-pose / YOLOv8 ──► protocolo do exercício · contagem · zonas ─────┐
 áudio ─┬► biomarcadores vocais ──► fadiga / disartria / respiratório ────────────┤
        ├► eventos AudioSet (heurística + AST) ──► tosse / respiração ────────────┤
        └► Azure Speech to Text ──► Azure Text Analytics ──► termos + sentimento ─┤
 vitais (PhysioNet) ──► regras NEWS2 · z robusto · CUSUM · Isolation Forest ──────┼─► FUSÃO ─► ALERTAS ─► equipe
                    └─► monitor em tempo real (streaming) ────────────────────────┤      │
 prescrições ──► farmacovigilância (regras + z-score) ────────────────────────────┤      └─► auditoria
 actigrafia ──► queda · imobilidade · agitação · Isolation Forest ────────────────┘
```

1. **Ingestão.** Cada modalidade tem um carregador próprio: registros PhysioNet a 1 Hz
   com PA arterial alinhada no tempo; WAV normalizado a 16 kHz mono (formato do Azure);
   vídeo lido quadro a quadro (OpenCV) ou keypoints/detecções pré-computados.
2. **Análise por modalidade.** Cada detector devolve uma lista de `Finding`
   (modalidade, tipo, severidade, mensagem, referência temporal, **método** e evidências)
   — o contrato comum que desacopla detectores, fusão e alertas.
3. **Fusão (late fusion).** O risco de cada modalidade é a severidade máxima dos seus
   achados (+0,03 por achado adicional). O risco fundido é
   `máx(modalidades) + 0,5·Σ(peso·risco das demais) + 0,1·nº de síndromes`, limitado a 1 —
   um achado crítico **nunca é diluído** por modalidades normais. Regras de **síndrome**
   exigem corroboração entre modalidades distintas (ex.: suspeita de AVC = disartria na voz
   *ou* relato neurológico **e** assimetria motora no vídeo).
4. **Alertas.** Cada achado vira um alerta priorizado (P1 crítico → time de resposta
   rápida; P2 alto → médico assistente e enfermagem; P3 moderado → painel), com
   deduplicação de 5 min por paciente/tipo, entrega por console, arquivo, painel web
   (Server-Sent Events) e webhook (Microsoft Teams / Azure Logic Apps), reconhecimento
   (ack) e trilha de auditoria.
5. **Tempo real.** O monitor de vitais processa amostra a amostra, só com o passado:
   regra clínica com **persistência de 10 s** (filtra artefato), **rearme** quando o sinal
   normaliza, **escalonamento** alto→crítico e z-score online (EWMA).

---

## 3. Dados utilizados

| Fonte | Natureza | Conteúdo |
|---|---|---|
| **PhysioNet BIDMC PPG and Respiration** (sugerido) | **real**, acesso aberto | 53 registros de 8 min de pacientes críticos (FC, pulso, FR, SpO₂ a 1 Hz). Registros 02, 32, 45, 25, 22 e 46 mapeados para PAC-0001…0006; os 47 restantes formam a população do Isolation Forest |
| **Google AudioSet** (sugerido) | **real** (metadados) | `ontology.json` (632 classes) + `eval_segments.csv` (20.371 segmentos) → catálogo de 22 classes clínicas com `mid`, descrição, contagem e exemplos de referência |
| **ESC-50** | **real** (Freesound, CC BY-NC) | 8 clipes de tosse/respiração — as classes correspondentes do AudioSet, cujo áudio não é distribuído. Selecionados ranqueando os 80 clipes pelo próprio classificador AudioSet (AST) |
| Consultas (áudio) | sintético controlado | Fala pt-BR sintetizada (voz Microsoft Maria, SSML) a partir de roteiros clínicos; perfis vocais por processamento de sinal (fadiga: lenta + energia decrescente; disartria: passa-baixa 2,4 kHz + tremor + instabilidade de F0 + pausas no meio da frase; respiratório: frases curtas + inspirações e tosses **reais**) |
| Vídeos | sintético controlado | Modelo cinemático COCO-17 (abdução de ombro, agachamento) e cena de cirurgia com detecções no formato YOLO, renderizados em H.264 |
| PA, prescrições, actigrafia | sintético controlado | Semente fixa; cada anomalia injetada é documentada (`pacientes.csv → cenario`) |

Mídia real de pacientes não pode ser versionada (LGPD) e não há dataset público com essas
anomalias rotuladas em português; por isso cada anomalia sintética é injetada de forma
**controlada e registrada**, o que permite medir precisão e recall (seção 5). O pipeline
aceita mídia real (`--file`), e a integração com o YOLOv8 real foi validada em imagens
públicas (seção 5.5).

---

## 4. Modelos aplicados em cada tipo de dado

### 4.1 Vídeo

**Postura (fisioterapia).** YOLOv8n-pose estima 17 keypoints COCO por quadro (pessoa de
maior bbox). O OpenPose sugerido no enunciado não tem build oficial para Python ≥ 3.12;
como o modelo COCO do OpenPose usa os mesmos pontos, `to_openpose18` converte para o layout
COCO-18 do OpenPose (pescoço = média dos ombros). Sobre a pose:

- **ângulos articulares**: abdução do ombro (braço × eixo do tronco), flexão do joelho,
  inclinação do tronco;
- **segmentação em repetições** (picos de amplitude) e comparação com o **protocolo**:
  ADM-alvo 150° (mínimo 60%), assimetria ≤ 20°, inclinação de tronco ≤ 10°,
  oscilação do tronco no agachamento ≤ 3°;
- **tremor**: energia 4–8 Hz da trajetória do punho normalizada pela largura dos ombros;
- **queda**: tronco > 60° da vertical com descida do quadril > 30% da altura em ≤ 1,5 s.

**Objetos e áreas críticas (cirurgia).** YOLOv8n (COCO: `scissors`, `knife`, `person`) ou
pesos *fine-tuned* de instrumental (`--weights`). Regras do checklist de Cirurgia Segura
(OMS) / AORN: contagem por classe na mesa no início × fechamento (modal em 1,5 s, robusta a
oclusão); item faltante localizado na área da incisão → *possível item retido*;
instrumento fora das zonas estéreis por ≥ 0,5 s → contaminação; pessoa não paramentada
sobre a mesa/campo → quebra de área estéril.

**Relatório automático.** Para cada vídeo, `reports/output/video/<vídeo>.md` com métricas,
tabela por repetição, desvios com instante e frames-chave anotados (esqueleto/bboxes).

### 4.2 Áudio

**Biomarcadores vocais** (numpy/scipy, frames de 25 ms): taxa de fala e de articulação
(núcleos silábicos), proporção de pausas e pausas longas/min, F0 por autocorrelação,
variabilidade de F0 em semitons, *jitter* e *shimmer* (proxies), HNR, energia de consoantes
(3–4,8 kHz vs. banda de voz), inclinação da energia (dB/min) e tremor (modulação 8–12 Hz).
Cada indicador é uma soma ponderada de rampas normal→alterado:

| Indicador | Marcadores (peso) |
|---|---|
| Fadiga | taxa de fala (0,25), pausas (0,20), queda de energia (0,30), pausas longas (0,15), frases curtas (0,10) |
| Disartria | tremor (0,30), energia de consoantes (0,30), articulação (0,15), jitter (0,10), frases curtas (0,15) |
| Respiratório | episódios de tosse (0,35), inspirações audíveis/min (0,35), frases curtas (0,15), pausas (0,15) |

As âncoras aceitam a **voz basal do próprio paciente** (`baseline_wav`).

**Eventos AudioSet.** Detector acústico nos intervalos entre falas (tosse: explosiva,
alta, banda larga; respiração: longa e de baixa energia), rotulado com o `mid` da
ontologia. Segunda opinião independente: **Audio Spectrogram Transformer**
(`MIT/ast-finetuned-audioset-10-10-0.4593`, mAP 0,459 no AudioSet) em janelas de 4 s,
filtrado para as classes do catálogo clínico.

**Azure Speech to Text** (REST de áudio curto, `pt-BR`, formato detalhado com confiança;
divisão em blocos ≤ 55 s nos silêncios) e **WER** contra a transcrição de referência.

**Azure Text Analytics**: `SentimentAnalysis` com opinion mining, `KeyPhraseExtraction`
e **Text Analytics for Health** (job assíncrono, entidades com asserção de negação). A
severidade operacional vem do **léxico clínico do hospital** (pt-BR, com detecção de
negação: "não sinto dor no peito" não dispara). Sem credenciais, o léxico assume o papel
completo e o resultado declara `provider = local_lexico`.

### 4.3 Sinais vitais

Quatro camadas, cada achado informa o método:

1. **Regras clínicas NEWS2** com persistência mínima de 10 s (FC 40/50/110/130,
   SpO₂ 88/92, FR 8/11/21/25, PAS 90/100/180/220);
2. **Z-score robusto** em janela móvel de 60 s (mediana/MAD, com piso de MAD) — mudanças
   abruptas mesmo dentro da faixa normal;
3. **CUSUM** unilateral — deriva sustentada (dessaturação progressiva, queda de PA);
4. **Isolation Forest populacional** (300 árvores) sobre janelas de 30 s
   (média/desvio/inclinação de FC, SpO₂, FR) treinado em 2.162 janelas de 47 registros
   PhysioNet; limiar no percentil 2 da população.

Episódios do mesmo sinal/método separados por < 30 s são **consolidados** (evita rajada de
alertas quando o sinal oscila no limiar).

### 4.4 Prescrições e movimentação

**Prescrições**: conflito de alergia por família farmacológica; salto de dose diária
≥ 2,5× (titulações usuais de 2× e desmames não disparam) ou corte ≤ 0,4× em classe
crítica; dose acima do máximo de referência; início de medicamento de alta vigilância
(ISMP); duplicidade de anticoagulante; suspensão de classe crítica sem substituto;
z-score de log-dose populacional.

**Movimentação** (actigrafia a cada 5 min por 48 h): queda (impacto ≥ 2,5 g + imobilidade),
imobilidade diurna ≥ 3 h sem reposicionamento, agitação noturna contínua ≥ 30 min
(z robusto vs. outras noites), Isolation Forest de perfis horários treinado em todos os
pacientes (só reporta horas não explicadas por regra).

---

## 5. Resultados obtidos

Todos os números abaixo foram gerados por `python -m src.main evaluate --audioset-model`
(`reports/avaliacao.md`) e `python -m src.main run-all` (`reports/resultados_execucao.md`).

### 5.1 Áudio

| Métrica | Resultado |
|---|---|
| Indicadores vocais (perfil injetado ⊆ flags) | **6/6 (100%)** — controles sem nenhuma flag |
| Tosse (episódios, tolerância 0,6 s) | precisão **0,80**, recall **1,00** (4 TP, 1 FP) |
| Respiração audível | precisão **1,00**, recall **0,45** (9 TP, 11 FN) |
| AST (AudioSet) — sons respiratórios | detectados em PAC-0002 (Gasp 0,45, Throat clearing 0,38) e PAC-0005 (Throat clearing 0,90); **nenhum** nos 3 pacientes sem eventos respiratórios; PAC-0004 com Gasp 0,14 (há uma inspiração injetada no perfil de fadiga) |

Indicadores observados (scores): PAC-0003 disartria 0,70 / PAC-0004 fadiga 0,76 /
PAC-0002 respiratório 0,88 / PAC-0005 fadiga 0,87 + respiratório 0,68 / controles 0,00.
Também disparou "fadiga" em PAC-0002 e PAC-0003 — coerente com o quadro (fala lenta e
entrecortada), mas registrado como achado extra.

**Limitação**: as respirações reais do ESC-50, mixadas a −12 dB da fala, ficam muitas
vezes abaixo do limiar do detector heurístico (recall 45%); o AST, porém, as reconhece
(Gasp/Snort). Uma tentativa de **sintetizar** tosse foi descartada porque o AST a
classificava como "Slap, smack" — o que motivou o uso de gravações reais.

### 5.2 Sinais vitais

**Injeção controlada** de 5 eventos (rampa de 5 s) no registro estável BIDMC 02:

| Evento | Lote | Tempo real | Latência |
|---|:-:|:-:|--:|
| SpO₂ → 85% (40 s) | ✅ regra + z + CUSUM | ✅ | 5 s |
| FC → 142 bpm (30 s) | ✅ regra + z + CUSUM | ✅ | 5 s |
| FR → 30 irpm (35 s) | ✅ regra + z + CUSUM | ✅ | 6 s |
| FC → 38 bpm (25 s) | ✅ regra + z | ✅ | 5 s |
| PAS → 78 mmHg (40 s) | ✅ regra + z + CUSUM | ✅ | 6 s |

Recall **100%** em lote e em tempo real, latência média **5,4 s** de sinal (persistência de
10 s contada a partir do primeiro valor fora da faixa durante a rampa), 1 falso positivo em
lote e **0** em tempo real fora das janelas injetadas.

**Dados reais**: BIDMC 32 (PAC-0002) — SpO₂ ≤ 87% nos 481 s (mín. 83%) e FR até 26 irpm,
ambos críticos, além de padrão multivariado atípico no Isolation Forest; BIDMC 25
(PAC-0004) — FC ≥ 110 bpm nos 481 s com pico de 139 bpm; BIDMC 22 (PAC-0005) — SpO₂ 91%
por 68 s; BIDMC 46 (PAC-0006) — FR 29 irpm (crítico) e derivas de FR/SpO₂. O controle
(BIDMC 02) não gerou nenhum achado.

### 5.3 Vídeo, prescrições, movimentação e fusão

| Modalidade | Recall dos achados esperados | Achados inesperados |
|---|--:|--:|
| Vídeo (4 vídeos, 9 achados esperados) | **100%** | 0 |
| Prescrições (6 pacientes, 8 esperados) | **100%** | 0 |
| Movimentação (6 pacientes, 3 esperados) | **100%** | 0 |
| Síndromes da fusão (8 esperadas) | **100%** | 0 |

Na fisioterapia de PAC-0003 a ADM medida (75° × 158°) coincide com a injetada; no
controle PAC-0001, 5 repetições com ADM 165°/162° e assimetria 3°, sem achados.

### 5.4 Risco fundido por paciente

| Paciente | Vitais | Áudio | Vídeo | Prescr. | Mov. | Risco | Síndromes |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| PAC-0001 controle | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | **0,00** baixo | — |
| PAC-0002 DPOC | 1,00 | 1,00 | — | 0,70 | 0,00 | **1,00** crítico | Insuficiência respiratória |
| PAC-0003 AVC | 0,40 | 1,00 | 0,79 | 0,00 | 0,70 | **1,00** crítico | Suspeita de AVC |
| PAC-0004 sepse | 1,00 | 0,79 | — | 1,00 | 0,00 | **1,00** crítico | Possível choque · Evento medicamentoso |
| PAC-0005 pneumonia | 0,76 | 1,00 | 0,98 | 0,00 | 0,98 | **1,00** crítico | Queda · Insuf. respiratória · Delirium |
| PAC-0006 pós-op. cardíaco | 1,00 | 0,95 | 1,00 | 0,76 | 0,00 | **1,00** crítico | Evento medicamentoso |

Observação: os cinco cenários de risco foram desenhados como críticos, então o risco
fundido satura em 1,00; a diferenciação entre eles está nas síndromes e nos achados.

### 5.5 Validação do YOLOv8 real

`scripts/validate_yolo.py` sobre imagens públicas da Ultralytics: `bus.jpg` — 4 pessoas,
16/17 keypoints confiáveis na principal, objetos `bus/person/stop sign`; `zidane.jpg` —
2 pessoas, objetos `person/tie`; conversão para OpenPose-18 e cálculo de ângulos funcionando
sobre pose real (`reports/output/yolo/`).

### 5.6 Exemplos de anomalias detectadas

| Modalidade | Exemplo (mensagem real gerada pelo sistema) |
|---|---|
| 🎥 Cirurgia | "Contagem final de instrumental não confere: 4/6 na mesa — faltam {'tesoura': 1, 'pinca': 1}; detectado(s): tesoura na área da incisão (possível item retido)" |
| 🎥 Cirurgia | "Pessoa não paramentada na área estéril entre 9.0s e 12.0s (3.0s)" |
| 🎥 Fisioterapia | "Assimetria média de 83° entre os braços (esq 75° × dir 158°)" · "QUEDA durante o exercício: tronco a 60° da vertical, quadril desceu 149px" |
| 🎙️ Voz | "Disartria (fala arrastada/imprecisa) detectada na voz (score 0.70)" |
| 🎙️ Texto | "Paciente relatou "dor no peito" (possivel sindrome coronariana)" — PAC-0006, voz normal |
| ❤️ Vitais | "Saturação de O₂ ↓ fora da faixa segura: 83 % por 481s" (PhysioNet real) |
| ❤️ Vitais | "Pressão sistólica ↓ deriva sustentada: 72 mmHg (basal 107) por 223s" |
| 💊 Prescrição | "piperacilina_tazobactam prescrito para paciente com alergia registrada a penicilina" |
| 💊 Prescrição | "Suspensão de heparina e enoxaparina deixa o paciente sem nenhum anticoagulante ativo" |
| 🛏️ Movimentação | "Provável QUEDA às 12/10 03:40: impacto de 3.6 g seguido de 30 min de imobilidade" |
| 🧩 Fusão | "Suspeita de AVC / déficit neurológico agudo — corroborado por 2 modalidade(s): Áudio/voz + texto, Vídeo → Acionar protocolo AVC" |

---

## 6. Integração com serviços Azure

| Serviço | Uso | Configuração |
|---|---|---|
| Azure AI Speech — Speech to Text | transcrição pt-BR das consultas, confiança, WER | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` |
| Azure AI Language — Sentiment / Key phrases | sentimento por frase e frases-chave | `AZURE_LANGUAGE_KEY`, `AZURE_LANGUAGE_ENDPOINT` |
| Azure AI Language — Text Analytics for Health | entidades clínicas com negação | idem |
| Teams Incoming Webhook / Azure Logic Apps | entrega de alertas P1/P2 ao plantonista | `ALERT_WEBHOOK_URL` |

O cliente é REST puro (`urllib`), sem SDK obrigatória. `python -m src.main azure-check`
valida credenciais e conectividade. Os resultados desta execução foram produzidos com os
**fallbacks locais** (transcrição de referência e léxico), pois as credenciais Azure não
estavam configuradas no ambiente de desenvolvimento; todo resultado registra o `provider`
efetivamente usado, inclusive na trilha de auditoria.

---

## 7. Segurança, ética e limitações

- **Apoio à decisão**: as ações sugeridas acionam protocolos e avaliação humana; nada é prescrito.
- **Fadiga de alarme**: persistência, consolidação de episódios, deduplicação e prioridade P1–P3.
- **Rastreabilidade**: cada avaliação e cada alerta (emissão, reconhecimento, falha de webhook) vai para `logs/audit_log.jsonl`.
- **Privacidade**: nenhum dado identificável; PhysioNet BIDMC é desidentificado e aberto.
- **Limitações**: vídeos, áudios e dados clínicos são sintéticos (a medida de desempenho
  vale para os cenários injetados, não para a população real); recall de respiração
  audível baixo; fusão por regras com pesos definidos pela equipe (não aprendidos); limiares
  de voz calibrados numa voz de síntese — num hospital seriam recalibrados com a voz basal
  do paciente; YOLOv8 COCO não reconhece pinça/porta-agulha sem fine-tuning.
- **Próximos passos**: validar com gravações reais anonimizadas e consentidas; YOLOv8
  fine-tuned em m2cai16-tool/Cholec80; fusão aprendida (ex.: regressão logística sobre os
  riscos por modalidade) com dados rotulados; Azure Event Hubs/Stream Analytics para
  ingestão de monitores reais.

---

## 8. Como reproduzir

```bash
pip install -r requirements.txt            # + requirements-ml.txt para YOLOv8/AST
python -m src.main run-all --audioset-model
python -m src.main evaluate --audioset-model
python scripts/validate_yolo.py
python -m pytest -q                         # 40 passed
python -m src.main serve --open             # painel da equipe médica
```
