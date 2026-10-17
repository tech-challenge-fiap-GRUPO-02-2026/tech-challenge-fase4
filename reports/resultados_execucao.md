# Resultados da execução — monitoramento multimodal

Gerado por `python -m src.main run-all`. Valores reais desta execução.

| Paciente | Cenário | Vitais | Áudio | Vídeo | Prescr. | Mov. | **Risco fundido** | Nível | Alertas |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| PAC-0001 | Pós-operatório de colecistectomia videolaparoscópica | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | baixo | 0 |
| PAC-0002 | DPOC exacerbada | 1.00 | 1.00 | 0.00 | 0.70 | 0.00 | **1.00** | critico | 10 |
| PAC-0003 | AVC isquêmico em reabilitação | 0.40 | 1.00 | 0.79 | 0.00 | 0.70 | **1.00** | critico | 12 |
| PAC-0004 | Sepse de foco urinário | 1.00 | 0.79 | 0.00 | 1.00 | 0.00 | **1.00** | critico | 17 |
| PAC-0005 | Pneumonia comunitária, alto risco de queda | 0.76 | 1.00 | 0.98 | 0.00 | 0.98 | **1.00** | critico | 17 |
| PAC-0006 | Pós-operatório de revascularização do miocárdio | 1.00 | 0.95 | 1.00 | 0.76 | 0.00 | **1.00** | critico | 12 |

## PAC-0001 — Pós-operatório de colecistectomia videolaparoscópica

_Cenário injetado:_ Controle — evolução esperada, sem anomalias injetadas

Nenhuma anomalia detectada.

## PAC-0002 — DPOC exacerbada

_Cenário injetado:_ Hipoxemia persistente real (BIDMC 32) + voz com sinais respiratórios + salto de corticoide

**Síndromes (fusão multimodal):**
- 🧩 **Desconforto/insuficiência respiratória — corroborado por 2 modalidade(s): Áudio/voz + texto, Sinais vitais** → Avaliação médica imediata da via aérea e oxigenação; considerar gasometria

| Severidade | Modalidade | Achado | Método |
|---|---|---|---|
| critico | vitals | Frequência respiratória ↑ fora da faixa segura: 26 irpm por 481s | regra_clinica_news2 |
| critico | vitals | Saturação de O₂ ↓ fora da faixa segura: 83 % por 481s | regra_clinica_news2 |
| critico | audio | Paciente relatou "falta de ar" (desconforto respiratorio) | local_lexico sobre a transcrição |
| alto | audio | Dificuldade respiratória detectada na voz (score 0.88) | biomarcadores acústicos (prosódia, F0, energia, eventos AudioSet) |
| alto | audio | Paciente relatou "chiado no peito" (broncoespasmo) | local_lexico sobre a transcrição |
| alto | prescriptions | Dose diária de metilprednisolona mudou 5.0× (80 → 400 mg/dia) por Dr. Souza | regra: variação ≥ 2.5× entre prescrições consecutivas |
| moderado | vitals | Padrão multivariado atípico vs. população PhysioNet (Frequência cardíaca média 81 bpm, Saturação de O₂ média 85 %, Frequência respiratória média 25 irpm) | isolation_forest_populacional |
| moderado | audio | Fadiga / cansaço vocal detectada na voz (score 0.61) | biomarcadores acústicos (prosódia, F0, energia, eventos AudioSet) |
| moderado | audio | 2 episódio(s) de tosse durante a consulta | detector acústico (classe AudioSet Cough /m/01b_21) |
| info | audio | Sentimento predominantemente negativo na consulta (100%) | local_lexico |

## PAC-0003 — AVC isquêmico em reabilitação

_Cenário injetado:_ FC basal baixa com episódios de bradicardia (BIDMC 45, 44–72 bpm) + disartria + assimetria na fisioterapia + imobilidade prolongada

**Síndromes (fusão multimodal):**
- 🧩 **Suspeita de AVC / déficit neurológico agudo — corroborado por 2 modalidade(s): Áudio/voz + texto, Vídeo** → Acionar protocolo AVC (código AVC): avaliação neurológica imediata e escala NIHSS

| Severidade | Modalidade | Achado | Método |
|---|---|---|---|
| critico | audio | Paciente relatou "fala está enrolada" (suspeita avc) | local_lexico sobre a transcrição |
| alto | audio | Paciente relatou "formigamento" (deficit neurologico) | local_lexico sobre a transcrição |
| alto | audio | Paciente relatou "engasgo" (disfagia risco aspiracao) | local_lexico sobre a transcrição |
| alto | video | ADM do ombro esquerdo 75° — 50% do alvo do protocolo (150°) | pose COCO-17 → ângulo ombro-cotovelo/tronco |
| alto | video | Assimetria média de 83° entre os braços (esq 75° × dir 158°) | comparação bilateral por repetição |
| alto | movement | 6.2 h sem reposicionamento no leito a partir de 11/10 11:45 (meta: mudança de decúbito a cada 2 h) | regra: ausência de picos de atividade em período diurno |
| moderado | vitals | Padrão multivariado atípico vs. população PhysioNet (Frequência cardíaca média 61 bpm, Saturação de O₂ média 100 %, Frequência respiratória média 15 irpm) | isolation_forest_populacional |
| moderado | audio | Fadiga / cansaço vocal detectada na voz (score 0.66) | biomarcadores acústicos (prosódia, F0, energia, eventos AudioSet) |
| moderado | audio | Disartria (fala arrastada/imprecisa) detectada na voz (score 0.70) | biomarcadores acústicos (prosódia, F0, energia, eventos AudioSet) |
| moderado | video | Compensação com inclinação lateral do tronco de até 15° durante a elevação | ângulo quadril→ombros vs. vertical |
| moderado | video | Tremor no punho esquerdo (índice 0.322, limite 0.2) | energia 4–8 Hz da trajetória do punho |
| info | audio | Sentimento predominantemente negativo na consulta (100%) | local_lexico |

## PAC-0004 — Sepse de foco urinário

_Cenário injetado:_ Taquicardia real (BIDMC 25) + hipotensão + fadiga vocal + prescrição com conflito de alergia

**Síndromes (fusão multimodal):**
- 🧩 **Possível choque (séptico/hemodinâmico) — corroborado por 1 modalidade(s): Sinais vitais** → Acionar time de resposta rápida; reavaliar perfusão e protocolo de sepse
- 🧩 **Risco de evento adverso medicamentoso — corroborado por 1 modalidade(s): Prescrições** → Farmácia clínica + médico prescritor revisarem a prescrição antes da próxima dose

| Severidade | Modalidade | Achado | Método |
|---|---|---|---|
| critico | vitals | Pressão sistólica ↓ fora da faixa segura: 72 mmHg por 210s | regra_clinica_news2 |
| critico | prescriptions | piperacilina_tazobactam prescrito para paciente com alergia registrada a penicilina | regra: família farmacológica × alergias do prontuário |
| critico | prescriptions | vancomicina: 5000 mg/dia acima do máximo de referência (4000 mg/dia) | regra: dose diária máxima |
| alto | vitals | Frequência cardíaca ↑ fora da faixa segura: 129 bpm por 481s | regra_clinica_news2 |
| alto | vitals | Pressão sistólica ↓ deriva sustentada: 72 mmHg (basal 107) por 223s | cusum_deriva |
| alto | vitals | Pressão diastólica ↓ fora da faixa segura: 40 mmHg por 194s | regra_clinica_news2 |
| alto | audio | Fadiga / cansaço vocal detectada na voz (score 0.76) | biomarcadores acústicos (prosódia, F0, energia, eventos AudioSet) |
| alto | audio | Paciente relatou "calafrios" (sinal infeccioso) | local_lexico sobre a transcrição |
| alto | audio | Paciente relatou "coração está acelerado" (taquicardia referida) | local_lexico sobre a transcrição |
| alto | prescriptions | Dose diária de vancomicina mudou 2.5× (2000 → 5000 mg/dia) por Dr. Souza | regra: variação ≥ 2.5× entre prescrições consecutivas |
| moderado | vitals | Frequência cardíaca ↑ mudança abrupta: 139 bpm por 6s | zscore_robusto_janela_60s |
| moderado | vitals | Padrão multivariado atípico vs. população PhysioNet (Frequência cardíaca média 120 bpm, Saturação de O₂ média 97 %, Frequência respiratória média 17 irpm) | isolation_forest_populacional |
| moderado | vitals | Frequência cardíaca ↓ mudança abrupta: 113 bpm por 14s | zscore_robusto_janela_60s |
| moderado | vitals | Frequência cardíaca ↓ mudança abrupta: 105 bpm por 16s | zscore_robusto_janela_60s |
| moderado | audio | 1 episódio(s) de tosse durante a consulta | detector acústico (classe AudioSet Cough /m/01b_21) |
| moderado | prescriptions | Início de medicamento de alta vigilância (noradrenalina) — exige dupla checagem | regra: lista ISMP de medicamentos potencialmente perigosos |
| info | audio | Sentimento predominantemente negativo na consulta (100%) | local_lexico |

## PAC-0005 — Pneumonia comunitária, alto risco de queda

_Cenário injetado:_ Dessaturações reais (BIDMC 22) + fadiga + perda de equilíbrio na fisio + agitação noturna e queda

**Síndromes (fusão multimodal):**
- 🧩 **Desconforto/insuficiência respiratória — corroborado por 2 modalidade(s): Áudio/voz + texto, Sinais vitais** → Avaliação médica imediata da via aérea e oxigenação; considerar gasometria
- 🧩 **Queda com possível lesão — corroborado por 3 modalidade(s): Áudio/voz + texto, Movimentação, Vídeo** → Avaliar o paciente no leito (trauma, nível de consciência); notificar evento adverso
- 🧩 **Risco de delirium — corroborado por 2 modalidade(s): Áudio/voz + texto, Movimentação** → Avaliar com escala CAM-ICU; revisar medicações e privação de sono

| Severidade | Modalidade | Achado | Método |
|---|---|---|---|
| critico | audio | Paciente relatou "falta de ar" (desconforto respiratorio) | local_lexico sobre a transcrição |
| critico | video | QUEDA durante o exercício: tronco a 60° da vertical, quadril desceu 149px | pose: ângulo do tronco + velocidade vertical do quadril |
| critico | movement | Provável QUEDA às 12/10 03:40: impacto de 3.6 g seguido de 30 min de imobilidade | regra: impacto ≥ 2,5 g + imobilidade subsequente |
| alto | vitals | Saturação de O₂ ↓ fora da faixa segura: 91 % por 68s | regra_clinica_news2 |
| alto | audio | Fadiga / cansaço vocal detectada na voz (score 0.87) | biomarcadores acústicos (prosódia, F0, energia, eventos AudioSet) |
| alto | audio | Paciente relatou "confusa" (delirium) | local_lexico sobre a transcrição |
| alto | audio | Paciente relatou "caí" (queda) | local_lexico sobre a transcrição |
| alto | audio | Paciente relatou "dor no quadril" (possivel fratura pos queda) | local_lexico sobre a transcrição |
| alto | video | Oscilação do tronco crescente (até 5.6°, limite 3°) — instabilidade/risco de queda | desvio-padrão da oscilação rápida do tronco |
| alto | movement | Agitação noturna: 165 min contínuos de atividade atípica a partir de 12/10 01:00, 3 saída(s) do leito — avaliar delirium | z-score robusto vs. outras noites do paciente |
| moderado | vitals | Padrão multivariado atípico vs. população PhysioNet (Frequência cardíaca média 85 bpm, Saturação de O₂ média 92 %, Frequência respiratória média 19 irpm) | isolation_forest_populacional |
| moderado | vitals | Frequência cardíaca ↑ mudança abrupta: 93 bpm por 7s | zscore_robusto_janela_60s |
| moderado | audio | Dificuldade respiratória detectada na voz (score 0.68) | biomarcadores acústicos (prosódia, F0, energia, eventos AudioSet) |
| moderado | audio | 2 episódio(s) de tosse durante a consulta | detector acústico (classe AudioSet Cough /m/01b_21) |
| info | audio | Sentimento predominantemente negativo na consulta (100%) | local_lexico |

## PAC-0006 — Pós-operatório de revascularização do miocárdio

_Cenário injetado:_ Taquipneia real (BIDMC 46) + falha de contagem de instrumental na cirurgia + suspensão abrupta de heparina

**Síndromes (fusão multimodal):**
- 🧩 **Risco de evento adverso medicamentoso — corroborado por 1 modalidade(s): Prescrições** → Farmácia clínica + médico prescritor revisarem a prescrição antes da próxima dose

| Severidade | Modalidade | Achado | Método |
|---|---|---|---|
| critico | vitals | Frequência respiratória ↑ fora da faixa segura: 29 irpm por 11s | regra_clinica_news2 |
| critico | audio | Paciente relatou "dor no peito" (possivel sindrome coronariana) | local_lexico sobre a transcrição |
| critico | video | Contagem final de instrumental não confere: 4/6 na mesa — faltam {'tesoura': 1, 'pinca': 1}; detectado(s): tesoura na área da incisão (possível item retido) | YOLO: contagem por classe na mesa (início × fechamento) |
| alto | vitals | Frequência respiratória ↑ fora da faixa segura: 25 irpm por 26s | regra_clinica_news2 |
| alto | vitals | Saturação de O₂ ↓ deriva sustentada: 92 % (basal 95) por 10s | cusum_deriva |
| alto | video | pinca saiu do campo estéril (provável queda) em 13.3s — considerar contaminado | YOLO: posição do instrumento vs. zonas estéreis |
| alto | video | Pessoa não paramentada na área estéril entre 9.0s e 12.0s (3.0s) | YOLO: pessoa × zonas estéreis |
| alto | prescriptions | enoxaparina iniciado com heparina ainda ativo (duplicidade de anticoagulante) | regra: duas drogas ativas da mesma classe crítica |
| alto | prescriptions | Suspensão de heparina e enoxaparina deixa o paciente sem nenhum anticoagulante ativo | regra: suspensão abrupta de classe crítica |
| moderado | vitals | Padrão multivariado atípico vs. população PhysioNet (Frequência cardíaca média 107 bpm, Saturação de O₂ média 94 %, Frequência respiratória média 23 irpm) | isolation_forest_populacional |
| moderado | vitals | Frequência respiratória ↑ deriva sustentada: 29 irpm (basal 18) por 49s | cusum_deriva |
| moderado | vitals | Padrão multivariado atípico vs. população PhysioNet (Frequência cardíaca média 107 bpm, Saturação de O₂ média 96 %, Frequência respiratória média 23 irpm) | isolation_forest_populacional |
| moderado | vitals | Frequência respiratória ↑ deriva sustentada: 25 irpm (basal 18) por 56s | cusum_deriva |
| moderado | prescriptions | Início de medicamento de alta vigilância (enoxaparina) — exige dupla checagem | regra: lista ISMP de medicamentos potencialmente perigosos |
