import json
import urllib.request
from pathlib import Path
from datetime import datetime

URL = "http://127.0.0.1:8081/v1/chat/completions"
MODEL = "/home/lasinac/Quimica/phi4_quimica_v2_BF16.gguf"
SKILL_PATH = Path("/home/lasinac/Quimica/skills/tutor-quimica-epn-SKILL.md")

if not SKILL_PATH.exists():
    raise FileNotFoundError(f"No se encontro la skill en: {SKILL_PATH}")

skill = SKILL_PATH.read_text(encoding="utf-8")

system_sin_skill = """
Eres un tutor academico de Quimica. Responde en espanol de forma clara y ordenada.
"""

tests = [
    {
        "id": "T1_kelvin_a_fahrenheit",
        "prompt": "El hidrógeno se vuelve líquido a 20,27 K. ¿Cuál es el equivalente en grados Fahrenheit?",
        "expected_terms": ["-423", "F"]
    },
    {
        "id": "T2_estado_oxidacion",
        "prompt": "Determine el estado de oxidación del nitrógeno en el compuesto N2O5.",
        "expected_terms": ["+5"]
    },
    {
        "id": "T3_estequiometria",
        "prompt": "Para la reacción 2H2 + O2 -> 2H2O, determine la masa de agua formada a partir de 4 g de H2 y 32 g de O2. Use H = 1 g/mol y O = 16 g/mol.",
        "expected_terms": ["36", "g"]
    }
]

required_headings = [
    "### 1. Fundamento químico del problema",
    "### 2. Información proporcionada",
    "### 3. Lo que se debe determinar",
    "### 4. Relaciones químicas y expresiones necesarias",
    "### 5. Aplicación de los datos",
    "### 6. Resolución paso a paso",
    "### 7. Resultado obtenido",
    "### 8. Conclusión"
]

forbidden_terms = [
    "Thought",
    "Solution",
    "<think>",
    "</think>",
    "```python",
    "import matplotlib",
    "plt."
]

def call_model(system_prompt, user_prompt):
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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

    return result["choices"][0]["message"]["content"]

def evaluate_response(text, expected_terms):
    score = 0
    details = []

    if text.strip().startswith("### 1. Fundamento químico del problema"):
        score += 2
        details.append("✅ Inicia con el encabezado obligatorio")
    else:
        details.append("❌ No inicia con el encabezado obligatorio")

    missing_headings = [h for h in required_headings if h not in text]
    if not missing_headings:
        score += 3
        details.append("✅ Incluye los 8 encabezados correctos")
    else:
        details.append("❌ Faltan encabezados: " + ", ".join(missing_headings))

    found_forbidden = [t for t in forbidden_terms if t in text]
    if not found_forbidden:
        score += 2
        details.append("✅ No incluye Thought, Solution, código ni etiquetas internas")
    else:
        details.append("❌ Incluye elementos prohibidos: " + ", ".join(found_forbidden))

    if all(term in text for term in expected_terms):
        score += 2
        details.append("✅ Contiene el resultado esperado")
    else:
        details.append("❌ No se detectó claramente el resultado esperado")

    if any(unit in text for unit in ["F", "g", "mol", "K", "^{\\circ}F", "°F"]):
        score += 1
        details.append("✅ Incluye unidades")
    else:
        details.append("❌ No se detectaron unidades claras")

    return score, details

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(f"/home/lasinac/Quimica/pruebas_skill_{timestamp}")
out_dir.mkdir(parents=True, exist_ok=True)

report = []
report.append("# Reporte comparativo: GGUF BF16 sin skill vs con skill\n")
report.append(f"Fecha: {datetime.now()}\n")
report.append(f"Modelo: {MODEL}\n")
report.append(f"Skill: {SKILL_PATH}\n\n")

for test in tests:
    print("=" * 80)
    print(f"Probando {test['id']} SIN skill...")
    respuesta_sin = call_model(system_sin_skill, test["prompt"])

    print(f"Probando {test['id']} CON skill...")
    respuesta_con = call_model(skill, test["prompt"])

    score_sin, details_sin = evaluate_response(respuesta_sin, test["expected_terms"])
    score_con, details_con = evaluate_response(respuesta_con, test["expected_terms"])

    file_sin = out_dir / f"{test['id']}_SIN_SKILL.md"
    file_con = out_dir / f"{test['id']}_CON_SKILL.md"

    file_sin.write_text(respuesta_sin, encoding="utf-8")
    file_con.write_text(respuesta_con, encoding="utf-8")

    report.append(f"## {test['id']}\n")
    report.append(f"**SIN skill:** {score_sin}/10\n")
    for d in details_sin:
        report.append(f"- {d}")
    report.append("")
    report.append(f"**CON skill:** {score_con}/10\n")
    for d in details_con:
        report.append(f"- {d}")
    report.append("\n---\n")

report_text = "\n".join(report)
report_path = out_dir / "reporte_comparativo_skill.md"
report_path.write_text(report_text, encoding="utf-8")

print("\n" + "=" * 80)
print("PRUEBA TERMINADA")
print("=" * 80)
print("Carpeta de resultados:")
print(out_dir)
print("\nReporte:")
print(report_path)
print("\nRESUMEN:")
print(report_text)
