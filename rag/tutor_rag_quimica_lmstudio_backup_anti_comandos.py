import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI


DB_PATH = "bd_semantica_quimica"
COLLECTION_NAME = "coleccion_libro_quimica_general"
REGISTRO_PATH = Path("registro_chunks_quimica.json")

LLM_BASE_URL = "http://127.0.0.1:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL = "phi4-quimica-v2"

SKILL_PATHS = [
    Path.home() / "Quimica" / "skills" / "tutor-quimica-epn-runtime.md",
    Path.home() / "Quimica" / "skills" / "tutor-quimica-epn-SKILL.md",
    Path.home() / "Quimica" / "skill" / "tutor-quimica-epn-runtime.md",
    Path.home() / "Quimica" / "skill" / "tutor-quimica-epn-SKILL.md",
]


def cargar_skill():
    for path in SKILL_PATHS:
        if path.exists():
            print(f"Skill cargada desde: {path}")
            return path.read_text(encoding="utf-8")

    print("No se encontró una skill en las rutas configuradas.")
    print("Se usará una instrucción básica de tutoría de Química.\n")

    return """
Eres un tutor académico especializado en Química General.
Responde con explicación clara, ordenada y paso a paso.
Cuando haya ejercicios numéricos, presenta datos, fórmula, sustitución, operación y resultado.
No inventes datos que no estén en el problema.
Usa un estilo pedagógico para estudiantes universitarios de primer nivel.
""".strip()


def cargar_bd():
    print("Cargando base semántica de Química...")

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    client_chroma = chromadb.PersistentClient(path=DB_PATH)

    collection = client_chroma.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    if not REGISTRO_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {REGISTRO_PATH}. Primero ejecuta crear_bd_semantica_quimica.py"
        )

    with open(REGISTRO_PATH, "r", encoding="utf-8") as f:
        registro = json.load(f)

    print("Base semántica cargada correctamente.\n")
    return collection, registro


def es_pregunta_conceptual(pregunta: str) -> bool:
    pregunta_limpia = pregunta.lower().strip()

    patrones_conceptuales = [
        "qué es",
        "que es",
        "define",
        "defina",
        "concepto",
        "explique qué",
        "explica qué",
        "explique que",
        "explica que",
        "diferencia",
        "diferencias",
        "importancia",
        "para qué sirve",
        "para que sirve",
        "cómo se calcula",
        "como se calcula",
        "cómo se calculan",
        "como se calculan",
        "qué diferencia",
        "que diferencia",
    ]

    patrones_ejercicio = [
        "calcule",
        "calcular",
        "determine",
        "determinar",
        "resuelva",
        "resolver",
        "halle",
        "hallar",
        "escriba la configuración",
        "escribe la configuración",
        "balancee",
        "balancear",
        "igualar",
        "convierta",
        "convertir",
        "cuántos gramos",
        "cuantos gramos",
        "cuántos moles",
        "cuantos moles",
    ]

    if any(patron in pregunta_limpia for patron in patrones_ejercicio):
        return False

    return any(patron in pregunta_limpia for patron in patrones_conceptuales)


def buscar_contexto(pregunta, collection, registro, n_resultados=3):
    resultados = collection.query(
        query_texts=[pregunta],
        n_results=n_resultados
    )

    contexto = []
    resumen_recuperacion = []

    for pos, chunk_id in enumerate(resultados["ids"][0], start=1):
        distancia = resultados["distances"][0][pos - 1]
        item = registro[chunk_id]

        resumen_recuperacion.append(
            f"{pos}. {chunk_id} | distancia={distancia:.4f} | "
            f"capítulo={item['capitulo']} | página={item['pagina']} | "
            f"tema={item['titulo_capitulo']}"
        )

        bloque = f"""
Fragmento recuperado {pos}
ID: {chunk_id}
Distancia semántica: {distancia:.4f}
Capítulo: {item["capitulo"]}
Título del capítulo: {item["titulo_capitulo"]}
Fuente: {item["fuente"]}
Página: {item["pagina"]}

Contenido del libro:
{item["contenido"]}
""".strip()

        contexto.append(bloque)

    separador = "\n\n" + "=" * 80 + "\n\n"
    contexto_final = separador.join(contexto)

    return contexto_final, resumen_recuperacion


def generar_respuesta_conceptual(pregunta, contexto, client_llm):
    system_final = """
Eres un tutor académico especializado en Química General.

La consulta fue clasificada como PREGUNTA CONCEPTUAL.

REGLAS OBLIGATORIAS:
- La respuesta debe iniciar exactamente con: ### Definición
- Está prohibido usar el formato de ocho apartados.
- No escribas encabezados numerados como ### 1., ### 2., ### 3.
- No escribas "Fundamento químico del problema".
- No escribas "Información proporcionada".
- No escribas "Lo que se debe determinar".
- No escribas "Relaciones químicas y expresiones necesarias".
- No escribas "Aplicación de los datos".
- No escribas "Resolución paso a paso".
- No escribas "Resultado obtenido".
- No escribas "Conclusión".
- No menciones fragmentos recuperados, capítulos, páginas, IDs ni distancias.
- No digas que la pregunta está escrita con minúsculas.
- No uses ejemplos de sodio ni configuración electrónica, salvo que la pregunta trate específicamente sobre configuración electrónica.
- Usa el contexto recuperado solo como apoyo teórico, no como parte visible de la respuesta.
- Responde de forma breve, clara, académica y pedagógica.

ESTRUCTURA ÚNICA PERMITIDA:
### Definición

### Explicación

### Ejemplo

### Importancia en Química
""".strip()

    prompt_usuario = f"""
Pregunta conceptual del estudiante:
{pregunta}

Contexto recuperado desde la base semántica:
{contexto}

Redacta la respuesta usando únicamente esta estructura:

### Definición
Define el concepto de forma directa.

### Explicación
Explica el concepto con mayor detalle.

### Ejemplo
Incluye un ejemplo breve y relacionado directamente con la pregunta.

### Importancia en Química
Explica por qué el concepto es importante.

No uses ocho apartados.
No uses numeración.
No menciones los fragmentos recuperados.
No uses ejemplo de sodio si la pregunta no trata sobre sodio o configuración electrónica.
""".strip()

    respuesta = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_final},
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=0.01,
        max_tokens=800
    )

    texto = respuesta.choices[0].message.content.strip()

    marcas_prohibidas = [
        "### 1.",
        "Fundamento químico del problema",
        "Información proporcionada",
        "Lo que se debe determinar",
        "Relaciones químicas y expresiones necesarias",
        "Aplicación de los datos",
        "Resolución paso a paso",
        "Resultado obtenido",
        "Conclusión",
        "Fragmentos recuperados",
        "Capítulo",
        "distancia="
    ]

    pregunta_lower = pregunta.lower()
    if "sodio" not in pregunta_lower and "configuración electrónica" not in pregunta_lower and "configuracion electronica" not in pregunta_lower:
        marcas_prohibidas.extend(["sodio", "Na:", "3s¹", "3s^1", "[Ne]"])

    if any(marca in texto for marca in marcas_prohibidas):
        reparacion = client_llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
Reescribe la respuesta como una explicación conceptual de Química General.

Está prohibido usar:
- Ocho apartados.
- Encabezados numerados.
- Fragmentos recuperados.
- Capítulos, páginas o distancias.
- Ejemplo de sodio, salvo que la pregunta lo solicite.

Usa únicamente:
### Definición
### Explicación
### Ejemplo
### Importancia en Química
""".strip()
                },
                {
                    "role": "user",
                    "content": f"""
Pregunta original:
{pregunta}

Respuesta incorrecta:
{texto}

Reescribe la respuesta correctamente usando únicamente esta estructura:

### Definición

### Explicación

### Ejemplo

### Importancia en Química
""".strip()
                }
            ],
            temperature=0.01,
            max_tokens=700
        )

        texto = reparacion.choices[0].message.content.strip()

    return texto

def generar_respuesta_ejercicio(pregunta, contexto, skill_pedagogica, client_llm):
    system_final = f"""
{skill_pedagogica}

MODO DE RESPUESTA: EJERCICIO DE QUÍMICA.

REGLAS OBLIGATORIAS:
1. Resuelve únicamente el ejercicio planteado por el estudiante.
2. Usa el contexto recuperado solo como apoyo conceptual.
3. No uses datos numéricos del contexto si no aparecen en la pregunta real.
4. La respuesta debe tener exactamente 8 apartados.
5. No omitas pasos importantes.
6. Si falta un dato necesario, indícalo claramente.
7. Mantén el estilo pedagógico de Química General.

ESTRUCTURA OBLIGATORIA:
### 1. Fundamento químico del problema
### 2. Información proporcionada
### 3. Lo que se debe determinar
### 4. Relaciones químicas y expresiones necesarias
### 5. Aplicación de los datos
### 6. Resolución paso a paso
### 7. Resultado obtenido
### 8. Conclusión
""".strip()

    prompt_usuario = f"""
PREGUNTA REAL DEL ESTUDIANTE:
{pregunta}

FRAGMENTOS RECUPERADOS DESDE EL LIBRO DE QUÍMICA:
{contexto}

INSTRUCCIÓN FINAL:
Resuelve el ejercicio con exactamente 8 apartados.
Usa únicamente los datos de la pregunta real del estudiante.
El contexto recuperado sirve solo como referencia teórica.
""".strip()

    respuesta = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_final},
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=0.05,
        max_tokens=3000
    )

    return respuesta.choices[0].message.content


def generar_respuesta(pregunta, collection, registro, skill_pedagogica, client_llm):
    contexto, resumen_recuperacion = buscar_contexto(
        pregunta=pregunta,
        collection=collection,
        registro=registro,
        n_resultados=3
    )

    print("Fragmentos recuperados:")
    for item in resumen_recuperacion:
        print(item)
    print()

    if es_pregunta_conceptual(pregunta):
        print("Modo detectado: pregunta conceptual\n")
        return generar_respuesta_conceptual(
            pregunta=pregunta,
            contexto=contexto,
            client_llm=client_llm
        )

    print("Modo detectado: ejercicio con 8 apartados\n")
    return generar_respuesta_ejercicio(
        pregunta=pregunta,
        contexto=contexto,
        skill_pedagogica=skill_pedagogica,
        client_llm=client_llm
    )


def main():
    skill_pedagogica = cargar_skill()
    collection, registro = cargar_bd()

    client_llm = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )

    print("Tutor RAG de Química General")
    print("Base semántica: bd_semantica_quimica")
    print("Colección: coleccion_libro_quimica_general")
    print("Modelo esperado: phi4_quimica_v2_BF16.gguf mediante LM Studio Server")
    print("Escribe 'salir' para terminar.\n")

    while True:
        pregunta = input("Pregunta: ").strip()

        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("Tutor finalizado.")
            break

        if not pregunta:
            continue

        print("\nBuscando fragmentos del libro y generando respuesta...\n")

        try:
            respuesta = generar_respuesta(
                pregunta=pregunta,
                collection=collection,
                registro=registro,
                skill_pedagogica=skill_pedagogica,
                client_llm=client_llm
            )

            print(respuesta)
            print("\n" + "=" * 100 + "\n")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
