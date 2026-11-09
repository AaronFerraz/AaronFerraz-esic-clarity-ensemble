"""Validação compartilhada dos arquivos de treino e entrega."""
from pathlib import Path
import pandas as pd

CLASSES = {"c1", "c234", "c5"}

def normalized_groups(texts):
    return (texts.fillna("").astype(str).str.normalize("NFKC")
            .str.lower().str.replace(r"\s+", " ", regex=True).str.strip())

def load_training(path):
    frame = pd.read_excel(path)
    missing = {"resp_text", "clarity"} - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    frame = frame.dropna(subset=["resp_text", "clarity"]).copy()
    texts = frame["resp_text"].astype(str)
    labels = frame["clarity"].astype(str)
    valid = texts.str.strip().ne("")
    texts, labels = texts[valid], labels[valid]
    if not len(texts) or set(labels) != CLASSES:
        raise ValueError("Treino deve ser não vazio e conter somente c1, c234 e c5, incluindo as três classes.")
    return texts.reset_index(drop=True), labels.reset_index(drop=True), normalized_groups(texts).reset_index(drop=True)

def load_test(path):
    test = pd.read_excel(path)
    missing = {"resp_text", "clarity"} - set(test.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no teste: {sorted(missing)}")
    if test["clarity"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("A coluna clarity do teste deve estar vazia.")
    return test

def validate_paths(train, test, destination):
    paths = [Path(p).resolve() for p in (train, test, destination)]
    if len(set(paths)) != 3:
        raise ValueError("Treino, teste e arquivo de saída precisam ter caminhos diferentes.")
