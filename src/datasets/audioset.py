"""Google AudioSet — ontologia e segmentos rotulados (research.google.com/audioset).

O AudioSet não distribui o áudio em si (são trechos de vídeos do YouTube),
e sim a **ontologia de 527 classes** e as listas de segmentos rotulados.
Usamos esses arquivos públicos para:

1. Montar o **catálogo de sons clinicamente relevantes** (tosse, respiração,
   chiado, arquejo, ronco...) com o ``mid`` oficial de cada classe — toda
   detecção acústica do sistema é rotulada com esse identificador, o que a
   torna compatível com qualquer classificador treinado no AudioSet
   (YAMNet, VGGish, PANNs, AST);
2. Medir a representatividade de cada classe no split de avaliação
   (``eval_segments.csv``) e listar exemplos de referência (YouTube IDs)
   para curadoria/validação humana;
3. Alimentar o classificador opcional ``MIT/ast-finetuned-audioset``
   (Audio Spectrogram Transformer treinado no AudioSet) em
   ``src/audio/audioset_classifier.py``.
"""
from __future__ import annotations

import csv
import json
import urllib.request
from collections import Counter

from src.config import AUDIOSET_DIR

URLS = {
    "ontology.json": "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json",
    "class_labels_indices.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
    "eval_segments.csv": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv",
}
CATALOG_PATH = AUDIOSET_DIR / "clinical_sound_catalog.json"

# Classes do AudioSet com relevância clínica → indicador do nosso sistema
CLINICAL_CLASSES = {
    "Cough": "respiratorio",
    "Breathing": "respiratorio",
    "Wheeze": "respiratorio",
    "Gasp": "respiratorio",
    "Pant": "respiratorio",
    "Snoring": "respiratorio",
    "Respiratory sounds": "respiratorio",
    "Throat clearing": "respiratorio",
    "Sneeze": "respiratorio",
    "Sniff": "respiratorio",
    "Snort": "respiratorio",
    "Sigh": "fadiga",
    "Groan": "dor",
    "Grunt": "dor",
    "Whimper": "dor",
    "Crying, sobbing": "sofrimento",
    "Hiccup": "outros",
    "Speech": "fala",
    "Male speech, man speaking": "fala",
    "Female speech, woman speaking": "fala",
    "Conversation": "fala",
    "Whispering": "fala",
}

PT_NAMES = {
    "Cough": "Tosse", "Breathing": "Respiração", "Wheeze": "Sibilância (chiado)",
    "Gasp": "Arquejo", "Pant": "Respiração ofegante", "Snoring": "Ronco",
    "Respiratory sounds": "Sons respiratórios", "Throat clearing": "Pigarro",
    "Sneeze": "Espirro", "Sniff": "Fungada", "Snort": "Bufo nasal", "Sigh": "Suspiro", "Groan": "Gemido",
    "Grunt": "Grunhido", "Whimper": "Choramingo", "Crying, sobbing": "Choro",
    "Hiccup": "Soluço", "Speech": "Fala", "Male speech, man speaking": "Fala masculina",
    "Female speech, woman speaking": "Fala feminina", "Conversation": "Conversa",
    "Whispering": "Sussurro",
}


def download(overwrite: bool = False) -> None:
    AUDIOSET_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = AUDIOSET_DIR / name
        if overwrite or not path.exists():
            urllib.request.urlretrieve(url, path)


def build_catalog(max_examples: int = 5) -> dict:
    ontology = json.loads((AUDIOSET_DIR / "ontology.json").read_text(encoding="utf-8"))
    by_name = {node["name"]: node for node in ontology}

    wanted = {by_name[name]["id"]: name for name in CLINICAL_CLASSES if name in by_name}
    counts: Counter = Counter()
    examples: dict[str, list[str]] = {mid: [] for mid in wanted}
    with (AUDIOSET_DIR / "eval_segments.csv").open(encoding="utf-8") as fh:
        reader = csv.reader(line for line in fh if not line.startswith("#"))
        for row in reader:
            ytid, start = row[0].strip(), row[1].strip()
            labels = ",".join(row[3:]).strip().strip('"').split(",")
            for mid in labels:
                mid = mid.strip()
                if mid in wanted:
                    counts[mid] += 1
                    if len(examples[mid]) < max_examples:
                        examples[mid].append(f"https://youtu.be/{ytid}?t={int(float(start))}")

    classes = []
    for mid, name in wanted.items():
        node = by_name[name]
        classes.append({
            "mid": mid,
            "name": name,
            "nome_pt": PT_NAMES.get(name, name),
            "indicador": CLINICAL_CLASSES[name],
            "descricao": node.get("description", ""),
            "segmentos_eval": counts[mid],
            "exemplos_referencia": examples[mid],
        })
    classes.sort(key=lambda c: (c["indicador"], -c["segmentos_eval"]))
    catalog = {
        "fonte": "Google AudioSet (ontology.json + eval_segments.csv)",
        "total_classes_ontologia": len(ontology),
        "classes_clinicas": classes,
    }
    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


def load_catalog() -> dict:
    if not CATALOG_PATH.exists():
        return build_catalog()
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def mid_for(name: str) -> str | None:
    for cls in load_catalog()["classes_clinicas"]:
        if cls["name"] == name:
            return cls["mid"]
    return None
