"""Paso 5: carga del adaptador entrenado, predicción por lotes y comparación contra el baseline del Módulo 3."""
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding


def cargar_con_adaptador(ckpt, dir_adapter, id2label, label2id):
    """Modelo base preentrenado (congelado) + adaptador LoRA entrenado en la Parte 2."""
    from peft import PeftModel
    base = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    return PeftModel.from_pretrained(base, str(dir_adapter))


@torch.no_grad()
def predecir(model, ds_tok, tokenizer, bs=128):
    """Devuelve (logits, segundos). Mide solo el forward del modelo, en el dispositivo disponible."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    cols = [c for c in ["input_ids", "attention_mask"] if c in ds_tok.column_names]
    loader = DataLoader(ds_tok.with_format("torch", columns=cols), batch_size=bs,
                        collate_fn=DataCollatorWithPadding(tokenizer))
    salidas = []
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for lote in loader:
        lote = {k: v.to(device) for k, v in lote.items() if k in cols}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            salidas.append(model(**lote).logits.float().cpu())
    if device == "cuda":
        torch.cuda.synchronize()
    return torch.cat(salidas).numpy(), time.time() - t0


def metricas(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": p, "recall_macro": r, "f1_macro": f,
            "f1_weighted": fw}


def mcnemar(acierto_a: np.ndarray, acierto_b: np.ndarray) -> dict:
    """Test de McNemar exacto (binomial) sobre los casos en que los modelos discrepan."""
    from scipy.stats import binomtest
    solo_a = int((acierto_a & ~acierto_b).sum())
    solo_b = int((~acierto_a & acierto_b).sum())
    n = solo_a + solo_b
    p = binomtest(solo_b, n, 0.5).pvalue if n else 1.0
    return {"solo_acierta_A": solo_a, "solo_acierta_B": solo_b, "p_valor": p}
