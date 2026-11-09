# DS III · Pre-entrega 4 — Clasificación con fine-tuning eficiente mediante LoRA

Proyecto integrador de **Data Science III: NLP & Deep Learning aplicado a Ciencia de Datos** (Coderhouse).
Autora: **Elizabeth Luque**

Clasificación de noticias **AG News** (Business, Sci_Tech, Sports, World) con un Transformer preentrenado ajustado mediante **LoRA (PEFT)**, comparado contra el baseline TF-IDF + Naive Bayes de la [Pre-entrega 3](https://github.com/scarletluque0805/ds3-clasificador-tfidf) (F1-macro 0,892 en test).

## Avance

| Parte | Pasos de la práctica | Estado |
|---|---|---|
| **1 · Datos, modelo base y configuración LoRA** | 1 · 2 · 3 | ✅ `notebooks/04_parte1_datos_modelo_lora.ipynb` |
| **2 · Fine-tuning en GPU y ablación** | 4 | 🔄 `notebooks/04_parte2_finetuning_lora.ipynb` listo para ejecutar en Colab (T4) |
| 3 · Benchmark contra el Módulo 3 y reporte PDF | 5 | ⏳ pendiente |

## Parte 1 · Decisiones

### Datos (Paso 1)
- Carga con `data/cargar_dataset.py` de la cátedra. `label` → entero en orden alfabético (`Business=0, Sci_Tech=1, Sports=2, World=3`) y `DatasetDict` de Hugging Face.
- **Splits:** train 7.200 / validation 800 (90/10 estratificado del train original) / **test 2.000 = el mismo del baseline del Módulo 3**.
- **Entrada al modelo:** solo la limpieza estructural del Módulo 2 (HTML, URLs, firmas de agencia, datelines), **sin lematizar ni quitar stop-words**, porque el Transformer necesita texto natural y su contexto.
- **`max_length = 72`**: p95 de sub-tokens WordPiece, el mismo valor del EDA del Módulo 2. Cubre el 95,7% de las noticias sin truncar, con padding dinámico.

### Modelo base (Paso 2): `distilbert-base-uncased`

| Modelo | Idioma | Capas | Parámetros | Entrenables con LoRA (r=8, Q-V) |
|---|---|---|---|---|
| **distilbert-base-uncased** ✅ | Inglés | 6 | 66,96 M | 741.124 (1,11%) |
| bert-base-uncased | Inglés | 12 | 109,49 M | 297.988 (0,27%) |
| roberta-base | Inglés | 12 | 124,65 M | 888.580 (0,71%) |
| distilbert-base-multilingual-cased | 104 idiomas | 6 | 135,33 M | 741.124 (0,55%) |

Corpus en **inglés** → modelo monolingüe en inglés (el multilingüe gasta el 68% de sus parámetros en embeddings de 104 idiomas). **6 capas** → la mitad de cómputo que BERT y RoBERTa, reteniendo alrededor del 97% de su desempeño. Es una tarea de comprensión de 4 temas con 7.200 ejemplos, y entra cómodo en una T4 gratuita. `roberta-base` queda como techo opcional.

### LoRA (Paso 3)

| Hiperparámetro | Valor | Por qué |
|---|---|---|
| `r` | 8 | tarea simple y cercana al preentrenamiento: rango intrínseco bajo (se valida con una ablación 4 / 8 / 16 en la Parte 2) |
| `lora_alpha` | 16 | escala α/r = 2 (convención estándar) |
| `target_modules` | `q_lin`, `v_lin` | Query y Value de la atención en las 6 capas: la mejor relación desempeño/parámetros según el paper de LoRA |
| `lora_dropout` | 0,1 | regulariza los adaptadores |
| learning rate (Parte 2) | 3e-4 | con el modelo base congelado no hay *catastrophic forgetting*; los adaptadores nuevos necesitan pasos más grandes que el 2e-5 del fine-tuning completo |

**Parámetros:** 741.124 entrenables de 66.956.548 (**1,11%**): 147.456 en adaptadores LoRA (0,22%) y 593.668 en el cabezal de clasificación nuevo. Se verificó que el Transformer base queda congelado y que las matrices B arrancan en cero.

## Parte 2 · Fine-tuning (Paso 4)

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scarletluque0805/ds3-transformer-lora/blob/main/notebooks/04_parte2_finetuning_lora.ipynb)

`Trainer` de Hugging Face con padding dinámico, `fp16` y evaluación por época, conservando la mejor época según el F1-macro de validación. Corridas:

| Corrida | r / α | learning rate | Objetivo |
|---|---|---|---|
| **principal** | 8 / 16 | 3e-4 | modelo de la entrega |
| lr_3e-5 | 8 / 16 | 3e-5 | mostrar que el lr del fine-tuning completo es bajo para LoRA |
| r4 · r16 | 4 / 8 · 16 / 32 | 3e-4 | validar la elección de `r` |
| roberta_r8 (opcional) | 8 / 16 | 3e-4 | techo de referencia con 12 capas |

Salidas: `reports/parte2/` (historial por época de cada corrida, resumen, figuras, entorno de GPU) y el adaptador LoRA en `models/lora_distilbert_r8/`. Las métricas se completan al ejecutar en GPU.

## Estructura

```
ds3-transformer-lora/
├── data/                 # cargar_dataset.py (cátedra) + ag_news/ (train y test originales)
├── src/
│   ├── limpieza.py       # limpieza estructural del Módulo 2 (regex, sin spaCy)
│   ├── datos_hf.py       # label -> int, splits y DatasetDict de Hugging Face
│   ├── modelos.py        # candidatos del Hub, LoraConfig y conteo de parámetros
│   └── entrenamiento.py  # tokenización, métricas, TrainingArguments y entrenamiento con LoRA
├── notebooks/
│   ├── 04_parte1_datos_modelo_lora.ipynb
│   └── 04_parte2_finetuning_lora.ipynb
├── reports/
│   ├── baseline_modulo3.json   # métricas del baseline TF-IDF (Pre-entrega 3)
│   ├── config_parte1.json      # configuración que usa la Parte 2
│   └── figures/
├── assets/               # vocabulario WordPiece (respaldo sin acceso al Hub)
└── requirements.txt
```

## Cómo ejecutarlo

En Google Colab con GPU (`Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`):

```python
!git clone https://github.com/scarletluque0805/ds3-transformer-lora.git
%cd ds3-transformer-lora
!pip install -q -r requirements.txt
```

y abrir `notebooks/04_parte1_datos_modelo_lora.ipynb`. La Parte 1 también corre en CPU.
