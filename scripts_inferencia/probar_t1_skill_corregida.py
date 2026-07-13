import json
import urllib.request
from pathlib import Path
from datetime import datetime

URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "/home/lasinac/Quimica/phi4_quimica_v2_BF16.gguf"
SKILL_PATH = Path("/home/lasinac/Quimica/skills/tutor-quimica-epn-SKILL.md")

skill = SKILL_PATH.read_text(encoding="utf-8")

prompt = "El hidrógeno se vuelve líquido a 20,27 K. ¿Cuál es el equivalente en grados Fahrenheit?"

data = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": skill},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0,
    "max_tokens": 1200
}

req = urllib.request.Request(
    URL,
    data=json.dumps(data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req, timeout=300) as response:
    result = json.loads(response.read().decode("utf-8"))

respuesta = result["choices"][0]["message"]["content"]

salida = Path("/home/lasinac/Quimica/prueba_t1_skill_corregida.md")
salida.write_text(respuesta, encoding="utf-8")

print("Respuesta guardada en:")
print(salida)
print("\n" + "=" * 80 + "\n")
print(respuesta)
