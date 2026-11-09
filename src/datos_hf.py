"""Paso 1: AG News (loader de la cátedra) -> etiquetas enteras -> DatasetDict de Hugging Face.

Splits:
  - train      : 90% del ag_news_train.csv (estratificado)  -> entrena los adaptadores LoRA
  - validation : 10% del ag_news_train.csv (estratificado)  -> curvas train vs. validación y mejor época
  - test       : ag_news_test.csv completo (2.000)           -> SOLO para el benchmark final contra el Módulo 3
El test es el mismo que usó el baseline TF-IDF, así la comparación es justa.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .limpieza import clean_text

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
sys.path.insert(0, str(ROOT / "data"))
from cargar_dataset import cargar  # noqa: E402  (loader provisto por la cátedra)


def mapeo_etiquetas(train_df: pd.DataFrame):
    """Mismo criterio que sugiere cargar_dataset.py: clases ordenadas alfabéticamente -> 0..n-1."""
    clases = sorted(train_df["label"].unique())
    label2id = {c: i for i, c in enumerate(clases)}
    id2label = {i: c for c, i in label2id.items()}
    return label2id, id2label


def preparar_dataframes(val_size: float = 0.10, seed: int = SEED):
    train_df, test_df = cargar("ag_news")
    label2id, id2label = mapeo_etiquetas(train_df)
    for df in (train_df, test_df):
        df["text"] = df["text"].map(clean_text)          # limpieza estructural del Módulo 2 (sin lematizar)
        df["labels"] = df["label"].map(label2id).astype("int64")   # el Trainer espera la columna 'labels' entera
    tr, va = train_test_split(train_df, test_size=val_size, stratify=train_df["labels"], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), test_df.reset_index(drop=True), label2id, id2label


def a_datasetdict(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame):
    from datasets import Dataset, DatasetDict
    cols = ["text", "labels"]
    return DatasetDict({
        "train": Dataset.from_pandas(tr[cols], preserve_index=False),
        "validation": Dataset.from_pandas(va[cols], preserve_index=False),
        "test": Dataset.from_pandas(te[cols], preserve_index=False),
    })
