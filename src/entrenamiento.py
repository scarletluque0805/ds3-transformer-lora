"""Paso 4: fine-tuning con LoRA usando el Trainer de Hugging Face."""
import inspect
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

from . import modelos as mo

SEED = 42


def tokenizar(ds, tokenizer, max_length: int = 72):
    """truncation=True con max_length (evita el error clásico de superar el límite) y SIN padding fijo:
    el DataCollatorWithPadding rellena cada lote solo hasta su noticia más larga (padding dinámico)."""
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=max_length), batched=True, remove_columns=["text"])


def compute_metrics(eval_pred):
    logits, y = eval_pred
    pred = np.argmax(logits, axis=-1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "precision_macro": p, "recall_macro": r, "f1_macro": f}


def cargar_preentrenado(ckpt, id2label, label2id):
    """Exige pesos preentrenados: entrenar LoRA sobre un modelo con pesos aleatorios no tendría sentido."""
    return AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=len(id2label), id2label=id2label, label2id=label2id)


def _training_args(out_dir, lr, epochs, bs, seed):
    kw = dict(output_dir=str(out_dir), learning_rate=lr, num_train_epochs=epochs,
              per_device_train_batch_size=bs, per_device_eval_batch_size=64, weight_decay=0.01, warmup_ratio=0.06,
              logging_strategy="epoch", save_strategy="epoch", save_total_limit=1,
              load_best_model_at_end=True, metric_for_best_model="f1_macro", greater_is_better=True,
              fp16=torch.cuda.is_available(), seed=seed, data_seed=seed, report_to="none")
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    return TrainingArguments(**{k: v for k, v in kw.items() if k in params})


def historial_por_epoca(log_history) -> pd.DataFrame:
    h = pd.DataFrame(log_history)
    tr = h[h.get("loss").notna()][["epoch", "loss"]].rename(columns={"loss": "train_loss"}) if "loss" in h else pd.DataFrame(columns=["epoch", "train_loss"])
    ev = h[h.get("eval_loss").notna()] if "eval_loss" in h else pd.DataFrame()
    cols = {"eval_loss": "val_loss", "eval_f1_macro": "val_f1_macro", "eval_accuracy": "val_accuracy"}
    ev = ev[["epoch", *cols]].rename(columns=cols)
    return pd.merge(tr, ev, on="epoch", how="inner").drop_duplicates("epoch").reset_index(drop=True)


def entrenar(nombre, ckpt, ds_tok, tokenizer, id2label, label2id, r=8, lora_alpha=None, lora_dropout=0.1,
             target_modules=None, lr=3e-4, epochs=4, bs=32, out_root="resultados", seed=SEED, base_model=None):
    """Entrena un modelo con LoRA y devuelve trainer, modelo, historial por época, tiempo y conteo de parámetros."""
    torch.manual_seed(seed)
    base = base_model if base_model is not None else cargar_preentrenado(ckpt, id2label, label2id)
    n_base = mo.contar(base)["total"]
    peft_model = mo.aplicar_lora(base, mo.config_lora(ckpt, r=r, lora_alpha=lora_alpha or 2 * r,
                                                      lora_dropout=lora_dropout, target_modules=target_modules))
    d = mo.desglose_entrenables(peft_model)
    trainer = Trainer(model=peft_model, args=_training_args(Path(out_root) / nombre, lr, epochs, bs, seed),
                      train_dataset=ds_tok["train"], eval_dataset=ds_tok["validation"],
                      data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    seg = time.time() - t0
    mejor = trainer.evaluate()   # con load_best_model_at_end: métricas de la mejor época en validación
    hist = historial_por_epoca(trainer.state.log_history)
    entrenables = d["adaptadores_lora"] + d["cabezal_clasificacion"]
    return dict(nombre=nombre, ckpt=ckpt, r=r, lora_alpha=lora_alpha or 2 * r, lr=lr, epochs=epochs,
                trainer=trainer, modelo=peft_model, historial=hist, segundos=seg,
                vram_pico_gb=(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
                params_base=n_base, params_entrenables=entrenables, params_lora=d["adaptadores_lora"],
                pct_entrenable=100 * entrenables / n_base,
                val_f1_macro=mejor["eval_f1_macro"], val_accuracy=mejor["eval_accuracy"], val_loss=mejor["eval_loss"],
                mejor_epoca=int(hist.loc[hist.val_f1_macro.idxmax(), "epoch"]) if len(hist) else None)


def fila_resumen(res: dict) -> dict:
    h = res["historial"]
    return {"corrida": res["nombre"], "modelo": res["ckpt"], "r": res["r"], "alpha": res["lora_alpha"], "lr": res["lr"],
            "% entrenable": round(res["pct_entrenable"], 2), "train_loss final": round(float(h.train_loss.iloc[-1]), 4),
            "val_loss final": round(float(h.val_loss.iloc[-1]), 4), "val F1-macro (mejor)": round(float(res["val_f1_macro"]), 4),
            "mejor época": res["mejor_epoca"], "minutos": round(res["segundos"] / 60, 2),
            "VRAM pico (GB)": round(res["vram_pico_gb"], 2) if res["vram_pico_gb"] else None}
