# Avaliação dos detectores

Gerado por `python -m src.main evaluate` — números reais desta execução.

## 1. Eventos acústicos (classes AudioSet) vs. eventos injetados

| Classe | TP | FP | FN | Precisão | Recall |
|---|--:|--:|--:|--:|--:|
| Cough | 4 | 1 | 0 | 0.80 | 1.00 |
| Breathing | 9 | 0 | 11 | 1.00 | 0.45 |

## 2. Indicadores vocais — acurácia 100%

| Paciente | Perfil injetado | Esperado | Detectado | Acerto |
|---|---|---|---|:-:|
| PAC-0001 | normal | — | — | ✅ |
| PAC-0002 | respiratorio | voz_respiratorio | voz_fadiga, voz_respiratorio | ✅ |
| PAC-0003 | disartria | voz_disartria | voz_disartria, voz_fadiga | ✅ |
| PAC-0004 | fadiga | voz_fadiga | voz_fadiga | ✅ |
| PAC-0005 | fadiga_respiratorio | voz_fadiga, voz_respiratorio | voz_fadiga, voz_respiratorio | ✅ |
| PAC-0006 | normal | — | — | ✅ |

### Classificador AST (AudioSet) — sons respiratórios

| Paciente | Perfil | Esperado respiratório | Classes detectadas (score máx) |
|---|---|:-:|---|
| PAC-0001 | normal | não | — |
| PAC-0002 | respiratorio | sim | Gasp 0.45, Snort 0.07, Sigh 0.17, Throat clearing 0.38, Sneeze 0.11 |
| PAC-0003 | disartria | não | — |
| PAC-0004 | fadiga | não | Gasp 0.14, Snort 0.07 |
| PAC-0005 | fadiga_respiratorio | sim | Gasp 0.29, Snort 0.11, Sneeze 0.38, Sigh 0.07, Throat clearing 0.90, Cough 0.07 |
| PAC-0006 | normal | não | — |

## 3. Sinais vitais — injeção controlada em PhysioNet BIDMC 02 (estável)

Recall em lote **100%**, em tempo real **100%**, latência média de disparo em tempo real **5.4 s** de sinal; falsos positivos fora das janelas injetadas: lote 1, tempo real 0.

| Sinal | Início | Alvo | Lote | Métodos | Tempo real | Latência |
|---|--:|--:|:-:|---|:-:|--:|
| SpO2 | 60s | 85.0 | ✅ | cusum_deriva, regra_clinica_news2, zscore_robusto_janela_60s | ✅ | 5.0s |
| HR | 150s | 142.0 | ✅ | cusum_deriva, regra_clinica_news2, zscore_robusto_janela_60s | ✅ | 5.0s |
| RESP | 230s | 30.0 | ✅ | cusum_deriva, regra_clinica_news2, zscore_robusto_janela_60s | ✅ | 6.0s |
| HR | 320s | 38.0 | ✅ | regra_clinica_news2, zscore_robusto_janela_60s | ✅ | 5.0s |
| SBP | 400s | 78.0 | ✅ | cusum_deriva, regra_clinica_news2, zscore_robusto_janela_60s | ✅ | 6.0s |

## 4. Achados esperados por paciente

| Modalidade | Recall | Achados inesperados |
|---|--:|--:|
| video | 100% | 0 |
| prescriptions | 100% | 0 |
| movement | 100% | 0 |
| fusion | 100% | 0 |

### video

| Paciente | Esperado | Detectado |
|---|---|---|
| PAC-0001 | — | — |
| PAC-0003 | adm_reduzida, assimetria, compensacao_tronco, tremor_punho_esquerdo | adm_reduzida_ombro_esquerdo, assimetria_membros_superiores, compensacao_tronco, tremor_punho_esquerdo |
| PAC-0005 | instabilidade_postural, queda_detectada | instabilidade_postural, queda_detectada |
| PAC-0006 | falha_contagem, instrumento_fora, quebra_area_esteril | falha_contagem_instrumental, instrumento_fora_campo_esteril, quebra_area_esteril |

### prescriptions

| Paciente | Esperado | Detectado |
|---|---|---|
| PAC-0001 | — | — |
| PAC-0002 | salto_de_dose | salto_de_dose |
| PAC-0003 | — | — |
| PAC-0004 | conflito_alergia, salto_de_dose, dose_acima_maxima, inicio_alta_vigilancia | conflito_alergia, salto_de_dose, dose_acima_maxima, inicio_alta_vigilancia |
| PAC-0005 | — | — |
| PAC-0006 | duplicidade_terapeutica, suspensao_anticoagulante, inicio_alta_vigilancia | inicio_alta_vigilancia, duplicidade_terapeutica, suspensao_anticoagulante_sem_substituto |

### movement

| Paciente | Esperado | Detectado |
|---|---|---|
| PAC-0001 | — | — |
| PAC-0002 | — | — |
| PAC-0003 | imobilidade_prolongada | imobilidade_prolongada |
| PAC-0004 | — | — |
| PAC-0005 | agitacao_noturna, queda_leito | agitacao_noturna, queda_leito |
| PAC-0006 | — | — |

### fusion

| Paciente | Esperado | Detectado |
|---|---|---|
| PAC-0001 | — | — |
| PAC-0002 | insuficiencia_respiratoria | insuficiencia_respiratoria |
| PAC-0003 | suspeita_avc | suspeita_avc |
| PAC-0004 | possivel_choque, risco_seguranca_medicamentosa | possivel_choque, risco_seguranca_medicamentosa |
| PAC-0005 | queda_confirmada, insuficiencia_respiratoria, delirium | insuficiencia_respiratoria, queda_confirmada, delirium |
| PAC-0006 | risco_seguranca_medicamentosa | risco_seguranca_medicamentosa |
