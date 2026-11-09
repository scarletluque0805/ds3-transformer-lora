"""Pasos 2 y 3: candidatos del Hugging Face Hub, configuración LoRA y auditoría de parámetros."""
import torch
from transformers import (AutoConfig, AutoModelForSequenceClassification, BertConfig, DistilBertConfig,
                          RobertaConfig)

# Candidatos evaluados. target_modules = nombre de las proyecciones Query y Value en cada arquitectura.
CANDIDATOS = {
    "distilbert-base-uncased": dict(familia="DistilBERT", idioma="Inglés", capas=6, target_modules=["q_lin", "v_lin"],
                                    config_offline=lambda: DistilBertConfig()),
    "bert-base-uncased": dict(familia="BERT", idioma="Inglés", capas=12, target_modules=["query", "value"],
                              config_offline=lambda: BertConfig()),
    "roberta-base": dict(familia="RoBERTa", idioma="Inglés", capas=12, target_modules=["query", "value"],
                         config_offline=lambda: RobertaConfig(vocab_size=50265, max_position_embeddings=514,
                                                              type_vocab_size=1, pad_token_id=1,
                                                              bos_token_id=0, eos_token_id=2)),
    "distilbert-base-multilingual-cased": dict(familia="DistilBERT", idioma="104 idiomas", capas=6,
                                               target_modules=["q_lin", "v_lin"],
                                               config_offline=lambda: DistilBertConfig(vocab_size=119547)),
}

MODELO_ELEGIDO = "distilbert-base-uncased"

# Configuración LoRA elegida (justificación en el notebook y en el README)
LORA_PARAMS = dict(r=8, lora_alpha=16, lora_dropout=0.1)


def cargar_base(ckpt: str, num_labels: int, id2label=None, label2id=None, pesos: bool = True):
    """Carga el modelo base. Con pesos=False (o sin acceso al Hub) arma la MISMA arquitectura desde su
    configuración, sin descargar pesos: sirve para contar parámetros, que no dependen de los valores."""
    kw = dict(num_labels=num_labels)
    if id2label:
        kw.update(id2label=id2label, label2id=label2id)
    if pesos:
        try:
            return AutoModelForSequenceClassification.from_pretrained(ckpt, **kw), "pesos preentrenados (Hub)"
        except Exception:
            pass
    try:
        cfg = AutoConfig.from_pretrained(ckpt, **kw)
        origen = "arquitectura del Hub (config.json), sin pesos"
    except Exception:
        cfg = CANDIDATOS[ckpt]["config_offline"]()
        for k, v in kw.items():
            setattr(cfg, k, v)
        origen = "arquitectura reconstruida localmente (config), sin pesos"
    torch.manual_seed(0)
    return AutoModelForSequenceClassification.from_config(cfg), origen


def config_lora(ckpt: str = MODELO_ELEGIDO, r=None, lora_alpha=None, lora_dropout=None, target_modules=None):
    from peft import LoraConfig, TaskType
    p = {**LORA_PARAMS}
    p.update({k: v for k, v in dict(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout).items() if v is not None})
    return LoraConfig(task_type=TaskType.SEQ_CLS, target_modules=target_modules or CANDIDATOS[ckpt]["target_modules"], **p)


def aplicar_lora(base_model, lora_cfg):
    from peft import get_peft_model
    return get_peft_model(base_model, lora_cfg)


def contar(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total=total, entrenables=entrenables, congelados=total - entrenables,
                pct_entrenable=100 * entrenables / total)


def desglose_entrenables(peft_model) -> dict:
    """Separa los parámetros entrenables en adaptadores LoRA (matrices A y B) y cabezal de clasificación."""
    lora = sum(p.numel() for n, p in peft_model.named_parameters() if p.requires_grad and "lora_" in n)
    cabezal = sum(p.numel() for n, p in peft_model.named_parameters() if p.requires_grad and "lora_" not in n)
    return dict(adaptadores_lora=lora, cabezal_clasificacion=cabezal)
