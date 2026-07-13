import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-4-reasoning"
ADAPTER_PATH = "/home/lasinac/Quimica/modelo_tutor_quimica_phi4_v2"

SYSTEM_PROMPT = """Eres un tutor academico de Quimica de la EPN.
Para todo ejercicio de quimica, responde obligatoriamente con estos 8 encabezados exactos:

### 1. Fundamento quimico del problema
### 2. Informacion proporcionada
### 3. Lo que se debe determinar
### 4. Relaciones quimicas y expresiones necesarias
### 5. Aplicacion de los datos
### 6. Resolucion paso a paso
### 7. Resultado obtenido
### 8. Conclusion

No escribas nada antes del primer encabezado.
No uses codigo Python.
No uses graficos.
No uses tablas.
Usa LaTeX con \\[ \\] o \\( \\).
Responde siempre en espanol academico.
"""

USER_PROMPT = """Resolver ejercicio de quimica:
El hidrógeno se vuelve líquido a 20,27 K. ¿Cuál es el equivalente en grados Fahrenheit?"""

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
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    max_memory={0: "31GiB", "cpu": "90GiB"}
)

print("Cargando adapter de Quimica v2...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    torch_dtype=torch.bfloat16
)

model.eval()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generando respuesta...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=850,
        do_sample=False,
        temperature=None,
        top_p=None,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id
    )

respuesta = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print("\n" + "=" * 80)
print(respuesta)
print("=" * 80)
