import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-4-reasoning"
ADAPTER_PATH = "./modelo_tutor_quimica_phi4_v2"
LOG_PATH = "pruebas_inferencia_phi4_v2.txt"

SYSTEM_PROMPT = """Eres un tutor académico experto en Química General de primer nivel universitario.
Resuelve los ejercicios de forma clara, ordenada y pedagógica.
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

def cargar_modelo():
    print("Cargando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Cargando modelo base en 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("Cargando adaptador QLoRA entrenado...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("Modelo cargado correctamente.\n")
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
        prompt = f"SYSTEM:\n{SYSTEM_PROMPT}\n\nUSER:\n{pregunta}\n\nASSISTANT:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=1400,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.05,
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
        f.write("PREGUNTA:\n")
        f.write(pregunta + "\n\n")
        f.write("RESPUESTA DEL MODELO:\n")
        f.write(respuesta + "\n")

def main():
    tokenizer, model = cargar_modelo()

    print("Prueba del modelo Phi-4 Química v2")
    print("Escribe un ejercicio de Química.")
    print("Para salir escribe: salir\n")

    while True:
        pregunta = input("Ejercicio > ").strip()

        if pregunta.lower() in ["salir", "exit", "q"]:
            print("Finalizando prueba.")
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
