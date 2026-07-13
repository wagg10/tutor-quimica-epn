"""
================================================================================
  PRUEBA DE INFERENCIA - TUTOR DE QUÍMICA QWEN2.5-14B v2
  Modelo base cargado en 8-bit + adaptador QLoRA entrenado
================================================================================
"""

import torch
import gc
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ═════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════

BASE_MODEL = "/home/lasinac/Quimica/modelos/Qwen2.5-14B-Instruct"
ADAPTER_PATH = "./modelo_tutor_quimica_qwen25_14b_v2"
LOG_PATH = "pruebas_inferencia_qwen25_14b_v2_8bit.txt"


SYSTEM_PROMPT = """Eres un tutor académico experto en Química General de primer nivel universitario.
Resuelve los ejercicios de forma clara, ordenada y pedagógica.

REGLA CRÍTICA DE ARITMÉTICA:
Antes de escribir cualquier resultado numérico final, SIEMPRE muestra
el cálculo paso a paso con números intermedios. Por ejemplo, en lugar
de escribir solo "9/5 × (-252,88) + 32 = -423,18", muestra:
  Paso A: 9/5 × (-252,88) = -455,184
  Paso B: -455,184 + 32 = -423,184
Esta verificación previene errores aritméticos.

Usa el siguiente formato cuando sea posible:
### 1. Fundamento químico del problema
### 2. Información proporcionada
### 3. Lo que se debe determinar
### 4. Relaciones químicas y expresiones necesarias
### 5. Aplicación de los datos
### 6. Resolución paso a paso
### 7. Resultado obtenido
### 8. Conclusión
"""


def limpiar_memoria():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def cargar_modelo():
    print("=" * 60)
    print("CARGANDO QWEN2.5-14B EN 8-BIT")
    print("=" * 60)

    limpiar_memoria()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU VRAM disponible: {free/1e9:.2f} GB / {total/1e9:.2f} GB")

    print("\n[1/3] Cargando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("\n[2/3] Cargando modelo base Qwen2.5-14B en 8-bit...")

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("\n[3/3] Aplicando adaptador QLoRA entrenado...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        used = total - free
        print(f"\nGPU VRAM en uso: {used/1e9:.2f} GB / {total/1e9:.2f} GB")

    print("\n✓ Modelo Qwen2.5-14B v2 cargado correctamente en 8-bit.\n")
    return tokenizer, model


def generar_respuesta(tokenizer, model, pregunta):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": pregunta},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = (
            f"SYSTEM:\n{SYSTEM_PROMPT}\n\n"
            f"USER:\n{pregunta}\n\n"
            f"ASSISTANT:\n"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1600,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    respuesta = tokenizer.decode(generated, skip_special_tokens=True)
    return respuesta.strip()


def guardar_prueba(pregunta, respuesta):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"FECHA: {datetime.now()}\n")
        f.write("MODELO:\n")
        f.write("Qwen2.5-14B-Instruct + adaptador QLoRA v2 en 8-bit\n\n")
        f.write("PREGUNTA:\n")
        f.write(pregunta + "\n\n")
        f.write("RESPUESTA DEL MODELO:\n")
        f.write(respuesta + "\n")


def main():
    tokenizer, model = cargar_modelo()

    print("=" * 60)
    print(" Prueba del modelo Qwen2.5-14B Química v2 8-bit")
    print("=" * 60)
    print("\nEscribe un ejercicio de Química.")
    print("Para salir escribe: salir\n")

    while True:
        pregunta = input("Ejercicio > ").strip()

        if pregunta.lower() in ["salir", "exit", "q"]:
            print("\nFinalizando prueba.")
            break

        if not pregunta:
            continue

        print("\nGenerando respuesta...\n")
        respuesta = generar_respuesta(tokenizer, model, pregunta)

        print("=" * 80)
        print(respuesta)
        print("=" * 80)

        guardar_prueba(pregunta, respuesta)
        print(f"\nPrueba guardada en: {LOG_PATH}\n")


if __name__ == "__main__":
    main()
