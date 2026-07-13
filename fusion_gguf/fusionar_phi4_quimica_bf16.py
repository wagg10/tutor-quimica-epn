import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

ADAPTER_PATH = os.path.expanduser(
    "~/Quimica/modelo_tutor_quimica_phi4_reasoning_qlora"
)

OUTPUT_PATH = os.path.expanduser(
    "~/Quimica/modelo_tutor_quimica_phi4_reasoning_merged_bf16"
)

print("=================================================")
print(" FUSIÓN PHI-4-REASONING + ADAPTADOR QLORA QUÍMICA")
print("=================================================")

if not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(f"No existe el adaptador: {ADAPTER_PATH}")

config_path = os.path.join(ADAPTER_PATH, "adapter_config.json")

if not os.path.exists(config_path):
    raise FileNotFoundError(f"No existe adapter_config.json en: {ADAPTER_PATH}")

with open(config_path, "r", encoding="utf-8") as f:
    adapter_config = json.load(f)

BASE_MODEL = adapter_config.get("base_model_name_or_path")

if not BASE_MODEL:
    raise ValueError("No se encontró 'base_model_name_or_path' en adapter_config.json")

print(f"Modelo base: {BASE_MODEL}")
print(f"Adaptador: {ADAPTER_PATH}")
print(f"Salida: {OUTPUT_PATH}")

# ============================================================
# TOKENIZER
# ============================================================

print("\nCargando tokenizer desde el MODELO BASE...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("pad_token no existía. Se usó eos_token como pad_token.")

print(f"Tamaño tokenizer: {len(tokenizer)}")
print(f"pad_token: {tokenizer.pad_token}")
print(f"eos_token: {tokenizer.eos_token}")

# ============================================================
# MODELO BASE EN BF16
# ============================================================

print("\nCargando modelo base en BF16 sobre CPU...")
print("Este paso puede tardar varios minutos.")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# ============================================================
# AJUSTE DE EMBEDDINGS
# ============================================================

print("\nVerificando tamaño de embeddings...")

embedding_size = base_model.get_input_embeddings().weight.shape[0]

print(f"Tamaño embeddings modelo: {embedding_size}")
print(f"Tamaño tokenizer: {len(tokenizer)}")

if embedding_size != len(tokenizer):
    print("Ajustando tamaño de embeddings...")
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.vocab_size = len(tokenizer)
else:
    print("No fue necesario ajustar embeddings.")

# ============================================================
# CARGA DEL ADAPTADOR
# ============================================================

print("\nCargando adaptador QLoRA de Química...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    is_trainable=False
)

# ============================================================
# FUSIÓN
# ============================================================

print("\nFusionando adaptador con el modelo base...")
print("Este paso puede tardar. No cierres la terminal.")

merged_model = model.merge_and_unload()

# ============================================================
# GUARDADO DEL MODELO FUSIONADO
# ============================================================

print("\nGuardando modelo fusionado en BF16...")

os.makedirs(OUTPUT_PATH, exist_ok=True)

merged_model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True,
    max_shard_size="4GB"
)

tokenizer.save_pretrained(OUTPUT_PATH)

print("\n✅ Fusión completada correctamente.")
print(f"Modelo fusionado guardado en: {OUTPUT_PATH}")
