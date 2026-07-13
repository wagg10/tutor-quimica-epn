"""
================================================================================
  PRUEBA DE INFERENCIA - TUTOR DE QUÍMICA PHI-4 v2
  Versión con bfloat16 para MÁXIMA PRECISIÓN aritmética
================================================================================

CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR:
  1. Modelo cargado en bfloat16 (16-bit) en lugar de 4-bit cuantizado
     → Mejora la precisión de cálculos numéricos
  2. System prompt fortalecido con instrucción de verificación aritmética
  3. repetition_penalty reducido de 1.05 a 1.0
     → Evita sesgos en repetición de números
  4. Mejor manejo de memoria GPU
================================================================================
"""

import torch
import gc
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ═════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════

BASE_MODEL = "microsoft/Phi-4-reasoning"
ADAPTER_PATH = "./modelo_tutor_quimica_phi4_v2"
LOG_PATH = "pruebas_inferencia_phi4_v2_bf16.txt"


# ─────────────────────────────────────────────
# System prompt fortalecido
# ─────────────────────────────────────────────
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


# ═════════════════════════════════════════════════════════════════
# CARGA DEL MODELO EN BFLOAT16
# ═════════════════════════════════════════════════════════════════

def cargar_modelo():
    print("=" * 60)
    print("CARGANDO MODELO EN BFLOAT16 (mayor precisión)")
    print("=" * 60)

    # Limpiar memoria GPU antes de cargar
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Mostrar memoria disponible
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

    print("\n[2/3] Cargando modelo base Phi-4 en bfloat16...")
    print("      (esto usa ~28 GB de VRAM en tu RTX 5090)")

    # CAMBIO CLAVE: bfloat16 en lugar de 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,  # ← Precisión de 16-bit
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("\n[3/3] Aplicando adaptador QLoRA entrenado...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Mostrar memoria final
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        used = total - free
        print(f"\nGPU VRAM en uso: {used/1e9:.2f} GB / {total/1e9:.2f} GB")

    print("\n✓ Modelo cargado correctamente en bfloat16.\n")
    return tokenizer, model


# ═════════════════════════════════════════════════════════════════
# GENERACIÓN DE RESPUESTAS
# ═════════════════════════════════════════════════════════════════

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
            max_new_tokens=1800,            # Aumentado para respuestas largas
            do_sample=False,                # Determinista
            temperature=None,
            top_p=None,
            repetition_penalty=1.0,         # CAMBIO: era 1.05, ahora sin penalización
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[-1]:]
    respuesta = tokenizer.decode(generated, skip_special_tokens=True)
    return respuesta.strip()


# ═════════════════════════════════════════════════════════════════
# GUARDAR PRUEBA
# ═════════════════════════════════════════════════════════════════

def guardar_prueba(pregunta, respuesta):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"FECHA: {datetime.now()}\n")
        f.write("PREGUNTA:\n")
        f.write(pregunta + "\n\n")
        f.write("RESPUESTA DEL MODELO:\n")
        f.write(respuesta + "\n")


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

def main():
    tokenizer, model = cargar_modelo()

    print("=" * 60)
    print(" Prueba del modelo Phi-4 Química v2 (bfloat16)")
    print(" Esta versión debería tener mejor precisión aritmética")
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
