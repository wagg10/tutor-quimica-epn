"""
================================================================================
  ENTRENAMIENTO QLORA - TUTOR ACADÉMICO DE QUÍMICA
  Escuela Politécnica Nacional | Proyecto TIC-CPGIC
  Modelo base: microsoft/Phi-4-reasoning
  Fine-Tuning con QLoRA 4-bit + PEFT
  
  VERSIÓN: 2.0 (Optimizada para dataset limpio de 684 ejercicios)
================================================================================

MEJORAS CLAVE EN ESTA VERSIÓN:
  1. Dataset corregido
  2. num_epochs aumentado a 4 (dataset más pequeño necesita más pasadas)
  3. eval_steps y save_steps ajustados al tamaño real del dataset
  4. learning_rate reducido a 1e-4 (más conservador, evita sobreajuste)
  5. max_seq_length aumentado a 3072 (los ejercicios con 8 secciones son largos)
  6. lora_dropout aumentado a 0.1 (más regularización)
  7. warmup_ratio aumentado a 0.05 (5% más estable con dataset pequeño)
  8. early_stopping con paciencia 8 (más tolerante)
  9. Validación del formato de 8 encabezados antes de entrenar
================================================================================
"""

import os
import gc
import json
import time
import random
import logging
import inspect
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    set_seed,
)

from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ═════════════════════════════════════════════════════════════════
# 0. CONFIGURACIÓN DEL SISTEMA
# ═════════════════════════════════════════════════════════════════

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"entrenamiento_qlora_deepseek_r1_qwen14b_v2_{timestamp}.log",
            encoding="utf-8",
        ),
    ],
)

log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN CENTRAL (OPTIMIZADA)
# ═════════════════════════════════════════════════════════════════

CFG = {
    # ── Paths ────────────────────────────────────────
    # CAMBIO 1: Apuntar al dataset corregido
    "dataset_path": "/home/lasinac/Quimica/dataset_quimica_final.jsonl",
    "output_dir": "./modelo_tutor_quimica_deepseek_r1_qwen14b_v2",
    "checkpoint_dir": "./checkpoints_deepseek_r1_qwen14b_v2",

    # ── Modelo ───────────────────────────────────────
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",

    # ── Reproducibilidad ─────────────────────────────
    "seed": 42,

    # ── Dataset ──────────────────────────────────────
    "test_size": 0.1,

    # ── QLoRA 4-bit ──────────────────────────────────
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.bfloat16,

    # ── LoRA (OPTIMIZADO) ────────────────────────────
    "lora_r": 64,
    "lora_alpha": 128,
    # CAMBIO 2: dropout aumentado para más regularización
    "lora_dropout": 0.1,
    "lora_target_modules": "all-linear",

    # ── Entrenamiento (OPTIMIZADO) ───────────────────
    # CAMBIO 3: max_seq_length aumentado (los ejercicios con 8 secciones son largos)
    "max_seq_length": 3072,
    "packing": False,

    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,

    # CAMBIO 4: 4 épocas (dataset pequeño necesita más pasadas para
    # aprender el formato de los 8 encabezados)
    "num_train_epochs": 4,

    # CAMBIO 5: learning rate reducido (más conservador, evita sobreajuste
    # en dataset limpio donde cada ejemplo es valioso)
    "learning_rate": 1e-4,

    "weight_decay": 0.01,

    # CAMBIO 6: warmup 5% (3% era muy poco con dataset pequeño)
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",

    # ── Evaluación / guardado (AJUSTADOS AL DATASET) ──
    # Con 615 ejemplos de train y batch efectivo 8 → 76 steps por época
    # CAMBIO 7: eval cada 19 steps = 4 evaluaciones por época
    "eval_steps": 19,
    # CAMBIO 8: save cada 38 steps = 2 checkpoints por época
    "save_steps": 38,
    "save_total_limit": 3,
    # CAMBIO 9: logging más frecuente para mejor visibilidad
    "logging_steps": 5,

    # ── Early stopping (MÁS TOLERANTE) ───────────────
    # CAMBIO 10: paciencia 8 (con eval cada 19 steps,
    # esto equivale a ~152 steps sin mejora)
    "early_stopping_patience": 8,
    "early_stopping_threshold": 0.001,

    # ── Hardware ─────────────────────────────────────
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",

    # ── Metadata de tesis ────────────────────────────
    "tesis_info": {
        "institucion": "Escuela Politécnica Nacional",
        "carrera": "Ingeniería de Software",
        "asignatura": "Química",
        "proyecto": "Implementación de Modelos Extensos de Lenguaje aplicados a "
                    "la tutoría académica en Química",
        "modelo_base": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "metodo": "QLoRA 4-bit + PEFT",
        "profesor": "PhD. Erick Herrera",
        "fecha": timestamp,
        "version_dataset": "v2 - 684 ejercicios corregidos con formato de 8 encabezados",
    },
}


# ═════════════════════════════════════════════════════════════════
# 2. SEMILLAS
# ═════════════════════════════════════════════════════════════════

def fijar_semillas(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


fijar_semillas(CFG["seed"])


# ═════════════════════════════════════════════════════════════════
# 3. UTILIDADES GPU
# ═════════════════════════════════════════════════════════════════

def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"disponible": False}

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)

    return {
        "disponible": True,
        "nombre": props.name,
        "vram_total_gb": round(total / 1e9, 2),
        "vram_libre_gb": round(free / 1e9, 2),
        "vram_usada_gb": round((total - free) / 1e9, 2),
        "compute_capability": f"{props.major}.{props.minor}",
    }


def limpiar_cache_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ═════════════════════════════════════════════════════════════════
# 4. VALIDACIÓN AVANZADA DEL DATASET
# ═════════════════════════════════════════════════════════════════

ENCABEZADOS_REQUERIDOS = [
    "### 1. Fundamento químico del problema",
    "### 2. Información proporcionada",
    "### 3. Lo que se debe determinar",
    "### 4. Relaciones químicas y expresiones necesarias",
    "### 5. Aplicación de los datos",
    "### 6. Resolución paso a paso",
    "### 7. Resultado obtenido",
    "### 8. Conclusión",
]

LATEX_PROHIBIDO = [
    r'\begin{table}',
    r'\begin{itemize}',
    r'\begin{equation}',
    r'\begin{tabular}',
]


def validar_dataset(ruta: str) -> None:
    """
    Validación EXHAUSTIVA del dataset:
      - Estructura JSON correcta
      - Presencia de los 8 encabezados markdown
      - Ausencia de LaTeX prohibido
      - Ausencia de código Python
    """
    log.info(f"Validando dataset en: {ruta}")

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe el dataset: {ruta}")

    errores_estructura = 0
    sin_8_encabezados = 0
    con_latex_prohibido = 0
    con_codigo_python = 0
    total = 0

    with open(ruta, "r", encoding="utf-8") as f:
        for i, linea in enumerate(f, 1):
            total += 1
            try:
                obj = json.loads(linea)

                # Estructura básica
                if "messages" not in obj or not isinstance(obj["messages"], list):
                    log.warning(f"Línea {i}: estructura messages inválida")
                    errores_estructura += 1
                    continue

                if len(obj["messages"]) < 3:
                    log.warning(f"Línea {i}: menos de 3 mensajes")
                    errores_estructura += 1
                    continue

                # Validar contenido del assistant
                assistant_msg = next(
                    (m for m in obj["messages"] if m.get("role") == "assistant"),
                    None
                )
                if not assistant_msg:
                    log.warning(f"Línea {i}: no hay mensaje 'assistant'")
                    errores_estructura += 1
                    continue

                contenido = assistant_msg.get("content", "")

                # Verificar 8 encabezados
                if not all(enc in contenido for enc in ENCABEZADOS_REQUERIDOS):
                    sin_8_encabezados += 1

                # Verificar LaTeX prohibido
                if any(cmd in contenido for cmd in LATEX_PROHIBIDO):
                    con_latex_prohibido += 1
                    log.warning(f"Línea {i}: LaTeX prohibido detectado")

                # Verificar código Python
                if 'import ' in contenido or 'def plot_' in contenido:
                    con_codigo_python += 1
                    log.warning(f"Línea {i}: código Python detectado")

            except json.JSONDecodeError as e:
                log.warning(f"Línea {i}: JSON inválido → {e}")
                errores_estructura += 1

    log.info("=" * 60)
    log.info("REPORTE DE VALIDACIÓN")
    log.info("=" * 60)
    log.info(f"Total ejemplos              : {total}")
    log.info(f"Errores de estructura       : {errores_estructura}")
    log.info(f"Sin 8 encabezados completos : {sin_8_encabezados}")
    log.info(f"Con LaTeX prohibido         : {con_latex_prohibido}")
    log.info(f"Con código Python           : {con_codigo_python}")
    log.info("=" * 60)

    if errores_estructura > 0 or con_latex_prohibido > 5 or con_codigo_python > 5:
        log.error("❌ Dataset con problemas críticos. Revisa antes de continuar.")
        raise ValueError("Dataset no apto para entrenamiento")

    log.info("✅ Dataset apto para entrenamiento")


# ═════════════════════════════════════════════════════════════════
# 5. CALLBACK LOGGING AVANZADO
# ═════════════════════════════════════════════════════════════════

class LoggingAvanzadoCallback(TrainerCallback):
    def __init__(self, ruta_historial: str):
        self.ruta = ruta_historial
        self.historial = []
        self.inicio_entrenamiento = time.time()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        elapsed = time.time() - self.inicio_entrenamiento

        registro = {
            "step": state.global_step,
            "epoch": round(state.epoch or 0, 3),
            "tiempo_min": round(elapsed / 60, 2),
            **metrics,
        }

        self.historial.append(registro)

        with open(self.ruta, "w", encoding="utf-8") as f:
            json.dump(self.historial, f, indent=2, ensure_ascii=False)

        eval_loss = metrics.get("eval_loss", None)
        eval_acc = metrics.get("eval_mean_token_accuracy", None)

        loss_txt = f"{eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else "N/A"
        acc_txt = f"{eval_acc:.4f}" if isinstance(eval_acc, (int, float)) else "N/A"

        log.info(
            f"📊 Eval step={state.global_step} | "
            f"epoch={registro['epoch']} | "
            f"loss={loss_txt} | "
            f"acc={acc_txt} | "
            f"t={registro['tiempo_min']} min"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Loggea también las métricas de entrenamiento."""
        if logs and "loss" in logs:
            log.info(
                f"📈 Train step={state.global_step} | "
                f"loss={logs['loss']:.4f} | "
                f"lr={logs.get('learning_rate', 0):.2e}"
            )

    def on_train_end(self, args, state, control, **kwargs):
        total = (time.time() - self.inicio_entrenamiento) / 60
        log.info(f"🏁 Entrenamiento completado en {total:.1f} minutos")


# ═════════════════════════════════════════════════════════════════
# 6. EARLY STOPPING
# ═════════════════════════════════════════════════════════════════

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 8, threshold: float = 0.001):
        self.patience = patience
        self.threshold = threshold
        self.mejor_loss = float("inf")
        self.contador = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        loss_actual = metrics.get("eval_loss", float("inf"))

        if loss_actual < self.mejor_loss - self.threshold:
            self.mejor_loss = loss_actual
            self.contador = 0
            log.info(f"✨ Nuevo mejor eval_loss: {self.mejor_loss:.4f}")
        else:
            self.contador += 1
            log.info(
                f"⏳ Sin mejora ({self.contador}/{self.patience}) | "
                f"Mejor: {self.mejor_loss:.4f} | Actual: {loss_actual:.4f}"
            )

            if self.contador >= self.patience:
                log.info("🛑 Early stopping activado")
                control.should_training_stop = True


# ═════════════════════════════════════════════════════════════════
# 7. MÉTRICAS
# ═════════════════════════════════════════════════════════════════

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    shifted_preds = predictions[:, :-1]
    shifted_labels = labels[:, 1:]

    mask = shifted_labels != -100
    total = mask.sum()

    if total == 0:
        return {"mean_token_accuracy": 0.0}

    correct = (shifted_preds == shifted_labels) & mask
    accuracy = float(correct.sum() / total)

    return {"mean_token_accuracy": accuracy}


# ═════════════════════════════════════════════════════════════════
# 8. PARÁMETROS
# ═════════════════════════════════════════════════════════════════

def contar_parametros(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "entrenables": entrenables,
        "porcentaje": round(100 * entrenables / total, 4),
    }


# ═════════════════════════════════════════════════════════════════
# 9. SFT CONFIG
# ═════════════════════════════════════════════════════════════════

def crear_sft_config() -> SFTConfig:
    """Compatible con versiones nuevas y viejas de TRL."""
    firma = inspect.signature(SFTConfig.__init__).parameters

    kwargs = {
        "output_dir": CFG["checkpoint_dir"],

        "per_device_train_batch_size": CFG["per_device_train_batch_size"],
        "per_device_eval_batch_size": CFG["per_device_eval_batch_size"],
        "gradient_accumulation_steps": CFG["gradient_accumulation_steps"],
        "gradient_checkpointing": CFG["gradient_checkpointing"],

        "learning_rate": CFG["learning_rate"],
        "weight_decay": CFG["weight_decay"],
        "warmup_ratio": CFG["warmup_ratio"],
        "lr_scheduler_type": CFG["lr_scheduler_type"],
        "optim": CFG["optim"],

        "num_train_epochs": CFG["num_train_epochs"],

        "fp16": CFG["fp16"],
        "bf16": CFG["bf16"],

        "logging_steps": CFG["logging_steps"],
        "report_to": "tensorboard",

        "save_strategy": "steps",
        "save_steps": CFG["save_steps"],
        "save_total_limit": CFG["save_total_limit"],

        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,

        "packing": CFG["packing"],
    }

    if "eval_strategy" in firma:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in firma:
        kwargs["evaluation_strategy"] = "steps"

    if "eval_steps" in firma:
        kwargs["eval_steps"] = CFG["eval_steps"]

    if "max_length" in firma:
        kwargs["max_length"] = CFG["max_seq_length"]
    elif "max_seq_length" in firma:
        kwargs["max_seq_length"] = CFG["max_seq_length"]

    if "gradient_checkpointing_kwargs" in firma:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if "include_num_input_tokens_seen" in firma:
        kwargs["include_num_input_tokens_seen"] = True

    kwargs_filtrados = {k: v for k, v in kwargs.items() if k in firma}

    omitidos = sorted(set(kwargs.keys()) - set(kwargs_filtrados.keys()))
    if omitidos:
        log.warning(f"Argumentos omitidos por compatibilidad: {omitidos}")

    return SFTConfig(**kwargs_filtrados)


# ═════════════════════════════════════════════════════════════════
# 10. MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 80)
    log.info(" TUTOR LLM QUÍMICA v2 - QLORA - DEEPSEEK-R1-DISTILL-QWEN 14B")
    log.info(f" Inicio: {timestamp}")
    log.info(" Dataset: 684 ejercicios corregidos con formato de 8 encabezados")
    log.info("=" * 80)

    gpu = gpu_info()
    log.info(f"GPU detectada: {gpu}")

    if not gpu["disponible"]:
        raise RuntimeError("No se detectó GPU CUDA. Revisa nvidia-smi.")

    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CFG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Guardar configuración ──────────────────────
    ruta_cfg = os.path.join(CFG["output_dir"], "configuracion_entrenamiento.json")
    cfg_guardable = {
        k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
        for k, v in CFG.items()
    }
    with open(ruta_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg_guardable, f, indent=2, ensure_ascii=False)
    log.info(f"Configuración guardada en: {ruta_cfg}")

    # ── Validación exhaustiva del dataset ──────────
    validar_dataset(CFG["dataset_path"])

    log.info(f"Cargando dataset desde: {CFG['dataset_path']}")
    dataset_crudo = load_dataset("json", data_files=CFG["dataset_path"])

    dataset = dataset_crudo["train"].train_test_split(
        test_size=CFG["test_size"],
        seed=CFG["seed"],
    )

    log.info(
        f"Dataset dividido → Train: {len(dataset['train'])} | "
        f"Validación: {len(dataset['test'])}"
    )

    # ── Calcular y mostrar plan de entrenamiento ───
    n_train = len(dataset['train'])
    batch_efectivo = (CFG['per_device_train_batch_size']
                      * CFG['gradient_accumulation_steps'])
    steps_por_epoca = max(1, n_train // batch_efectivo)
    total_steps = steps_por_epoca * CFG['num_train_epochs']
    warmup_steps = int(total_steps * CFG['warmup_ratio'])

    log.info("=" * 60)
    log.info("PLAN DE ENTRENAMIENTO")
    log.info("=" * 60)
    log.info(f"  Batch efectivo            : {batch_efectivo}")
    log.info(f"  Steps por época           : {steps_por_epoca}")
    log.info(f"  Total steps               : {total_steps}")
    log.info(f"  Warmup steps              : {warmup_steps}")
    log.info(f"  Evaluaciones por época    : ~{steps_por_epoca // CFG['eval_steps']}")
    log.info(f"  Checkpoints por época     : ~{steps_por_epoca // CFG['save_steps']}")
    log.info("=" * 60)

    # ── Tokenizer ──────────────────────────────────
    log.info(f"Cargando tokenizador: {CFG['model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(
        CFG["model_id"],
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("pad_token = eos_token")

    # ── QLoRA 4-bit config ─────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CFG["load_in_4bit"],
        bnb_4bit_quant_type=CFG["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=CFG["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=CFG["bnb_4bit_compute_dtype"],
    )

    # ── Modelo ─────────────────────────────────────
    log.info(f"Cargando modelo base en QLoRA 4-bit: {CFG['model_id']}")
    limpiar_cache_gpu()

    model = AutoModelForCausalLM.from_pretrained(
        CFG["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=CFG["gradient_checkpointing"],
    )

    log.info(f"GPU tras cargar modelo: {gpu_info()}")

    # ── LoRA config ────────────────────────────────
    peft_config = LoraConfig(
        r=CFG["lora_r"],
        lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"],
        target_modules=CFG["lora_target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )

    # ── Formato chat ───────────────────────────────
    def format_chat_template(example: dict) -> str:
        try:
            return tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            texto = ""
            for msg in example["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                texto += f"{role.upper()}:\n{content}\n\n"
            return texto + tokenizer.eos_token

    # ── SFTConfig ──────────────────────────────────
    sft_config = crear_sft_config()

    # ── Callbacks ──────────────────────────────────
    ruta_historial = os.path.join(CFG["output_dir"], "historial_metricas.json")
    callbacks = [
        LoggingAvanzadoCallback(ruta_historial=ruta_historial),
        EarlyStoppingCallback(
            patience=CFG["early_stopping_patience"],
            threshold=CFG["early_stopping_threshold"],
        ),
    ]

    # ── Trainer ────────────────────────────────────
    firma_trainer = inspect.signature(SFTTrainer.__init__).parameters

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["test"],
        "args": sft_config,
        "peft_config": peft_config,
        "formatting_func": format_chat_template,
        "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }

    if "processing_class" in firma_trainer:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    params = contar_parametros(trainer.model)
    log.info(
        f"Parámetros QLoRA → Total: {params['total']:,} | "
        f"Entrenables: {params['entrenables']:,} | "
        f"{params['porcentaje']}%"
    )

    # ── Entrenar ───────────────────────────────────
    limpiar_cache_gpu()
    log.info("\n🚀 INICIANDO ENTRENAMIENTO QLORA v2 🚀\n")
    t0 = time.time()

    resultado = trainer.train()

    duracion_total = (time.time() - t0) / 60
    log.info(f"⏱ Duración total: {duracion_total:.1f} minutos")

    # ── Evaluación final ───────────────────────────
    log.info("\n📊 Evaluación final...")
    metricas_finales = trainer.evaluate()
    log.info(f"Métricas finales: {metricas_finales}")

    log.info(f"Mejor checkpoint: {trainer.state.best_model_checkpoint}")
    log.info(f"Mejor métrica: {trainer.state.best_metric}")

    # ── Guardar adaptador ──────────────────────────
    log.info(f"\n💾 Guardando adaptador QLoRA en: {CFG['output_dir']}")
    trainer.model.save_pretrained(CFG["output_dir"])
    tokenizer.save_pretrained(CFG["output_dir"])

    # ── Reporte ────────────────────────────────────
    reporte = {
        "metadata_tesis": CFG["tesis_info"],
        "configuracion": cfg_guardable,
        "gpu_info": gpu,
        "parametros_lora": params,
        "plan_entrenamiento": {
            "batch_efectivo": batch_efectivo,
            "steps_por_epoca": steps_por_epoca,
            "total_steps_planificados": total_steps,
            "warmup_steps": warmup_steps,
        },
        "resultado_train": {
            "global_step": resultado.global_step,
            "training_loss": resultado.training_loss,
            "duracion_minutos": round(duracion_total, 2),
        },
        "mejor_checkpoint": trainer.state.best_model_checkpoint,
        "mejor_metrica_valor": trainer.state.best_metric,
        "metricas_finales": metricas_finales,
    }

    ruta_reporte = os.path.join(CFG["output_dir"], "reporte_entrenamiento.json")
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    log.info("=" * 80)
    log.info("✅ ENTRENAMIENTO QLORA DEEPSEEK-R1-QWEN14B v2 COMPLETADO")
    log.info(f"Modelo guardado: {CFG['output_dir']}")
    log.info(f"Reporte:         {ruta_reporte}")
    log.info(f"Historial:       {ruta_historial}")
    log.info(f"Log:             entrenamiento_qlora_deepseek_r1_qwen14b_v2_{timestamp}.log")
    log.info("=" * 80)

    log.info("\n📋 PRÓXIMOS PASOS:")
    log.info("  1. Convertir el adaptador LoRA a GGUF:")
    log.info(f"     python convert_lora_to_gguf.py --adapter {CFG['output_dir']}")
    log.info("  2. Combinar con el modelo base Phi-4 y cuantizar a Q4_K_M")
    log.info("  3. Reemplazar el archivo phi4_quimica_Q4_K_M.gguf en Hermes")
    log.info("  4. Reiniciar llama-server y probar con un ejercicio")
    log.info("")

    return reporte


if __name__ == "__main__":
    main()