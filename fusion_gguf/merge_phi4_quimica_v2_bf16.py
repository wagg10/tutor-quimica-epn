import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-4-reasoning"
ADAPTER_PATH = "/home/lasinac/Quimica/modelo_tutor_quimica_phi4_v2"
OUT_DIR = "/home/lasinac/Quimica/modelo_tutor_quimica_phi4_v2_merged_bf16_clean"

print("Limpiando memoria CUDA...")
torch.cuda.empty_cache()

print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

print("Cargando modelo base en BF16...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    max_memory={0: "31GiB", "cpu": "100GiB"}
)

print("Cargando adapter de Quimica v2...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    dtype=torch.bfloat16
)

print("Fusionando adapter con modelo base...")
merged_model = model.merge_and_unload()

print("Guardando modelo fusionado BF16 limpio...")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

merged_model.save_pretrained(
    OUT_DIR,
    safe_serialization=True,
    max_shard_size="4GB"
)

tokenizer.save_pretrained(OUT_DIR)

print("Modelo fusionado guardado en:")
print(OUT_DIR)
