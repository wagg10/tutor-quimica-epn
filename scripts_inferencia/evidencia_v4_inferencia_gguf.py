import json
import urllib.request

url = "http://127.0.0.1:8081/v1/chat/completions"

prompt = """Resolver ejercicio de quimica.

Usa exactamente estos 8 apartados:
### 1. Fundamento quimico del problema
### 2. Informacion proporcionada
### 3. Lo que se debe determinar
### 4. Relaciones quimicas y expresiones necesarias
### 5. Aplicacion de los datos
### 6. Resolucion paso a paso
### 7. Resultado obtenido
### 8. Conclusion

No uses Python, graficos ni tablas.

Ejercicio:
El hidrógeno se vuelve líquido a 20,27 K. ¿Cuál es el equivalente en grados Fahrenheit?

Datos verificados:
20,27 - 273,15 = -252,88
1,8 × (-252,88) = -455,184
-455,184 + 32 = -423,184
Resultado final = -423,18 °F
"""

data = {
    "model": "/home/lasinac/Quimica/phi4_quimica_v2_BF16.gguf",
    "messages": [
        {
            "role": "system",
            "content": "Eres un tutor academico de Quimica. Responde en espanol, con explicacion paso a paso y formato academico."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    "temperature": 0,
    "max_tokens": 900
}

req = urllib.request.Request(
    url,
    data=json.dumps(data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urllib.request.urlopen(req) as response:
    result = json.loads(response.read().decode("utf-8"))

respuesta = result["choices"][0]["message"]["content"]

print("EVIDENCIA V.4 - PRUEBA DE INFERENCIA LOCAL")
print("Modelo: phi4_quimica_v2_BF16.gguf")
print("Endpoint: http://127.0.0.1:8081/v1")
print("=" * 80)
print(respuesta)
