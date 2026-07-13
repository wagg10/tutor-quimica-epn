import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI


DB_PATH = "bd_semantica_quimica"
COLLECTION_NAME = "coleccion_libro_quimica_general"
REGISTRO_PATH = Path("registro_chunks_quimica.json")

LLM_BASE_URL = "http://127.0.0.1:8081/v1"
LLM_API_KEY = "local"

# En llama-server el nombre puede variar.
# Este nombre debe funcionar como identificador local.
LLM_MODEL = "phi4_quimica_v2_BF16"

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


def buscar_contexto(pregunta, collection, registro, n_resultados=5):
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


def es_pregunta_conceptual(pregunta: str) -> bool:
    pregunta_limpia = pregunta.lower().strip()

    patrones_conceptuales = [
        "qué es",
        "que es",
        "define",
        "defina",
        "explica",
        "explique",
        "diferencia",
        "diferencias",
        "concepto",
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

    return any(patron in pregunta_limpia for patron in patrones_conceptuales)


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

    conceptual = es_pregunta_conceptual(pregunta)

    if conceptual:
        system_final = """
Eres un tutor académico especializado en Química General.

REGLAS PARA PREGUNTAS CONCEPTUALES:
1. Responde de forma clara, completa y pedagógica.
2. No uses el formato de 8 apartados.
3. No escribas secciones como "Fundamento químico", "Información proporcionada", "Resolución paso a paso" ni "Resultado obtenido".
4. No conviertas la pregunta conceptual en un ejercicio numérico.
5. Sí debes incluir un ejemplo sencillo cuando ayude a comprender el concepto.
6. El ejemplo debe ser breve y debe decir claramente "Ejemplo".
7. No inventes datos complejos ni resuelvas un problema largo si el estudiante no lo pidió.
8. Usa el contexto recuperado como apoyo teórico, pero no lo copies literalmente.
9. La respuesta debe ser más desarrollada que una definición simple.
10. Mantén un estilo académico adecuado para Química General universitaria.

ESTRUCTURA PARA PREGUNTAS CONCEPTUALES:
### Definición
Explica el concepto de forma directa.

### Explicación
Desarrolla la idea con mayor detalle.

### Ejemplo
Incluye un ejemplo sencillo relacionado con el concepto.

### Importancia en Química
Explica para qué sirve o por qué es importante.
""".strip()

        prompt_usuario = f"""
PREGUNTA REAL DEL ESTUDIANTE:
{pregunta}

FRAGMENTOS RECUPERADOS DESDE EL LIBRO DE QUÍMICA:
{contexto}

INSTRUCCIÓN:
Responde la pregunta conceptual con una explicación completa y un ejemplo sencillo.
No uses formato de ejercicio.
No uses los 8 apartados.
No agregues ejercicios largos si el estudiante solo pide una explicación conceptual.
""".strip()

        max_tokens = 1500
        temperature = 0.03

    else:
        system_final = f"""
{skill_pedagogica}

REGLAS PARA EJERCICIOS DE QUÍMICA:
1. Resuelve únicamente el ejercicio planteado por el estudiante.
2. Usa el contexto recuperado solo como apoyo conceptual.
3. No uses datos numéricos del contexto si no aparecen en la pregunta real.
4. La respuesta debe tener exactamente 8 apartados.
5. Mantén el estilo de tutor académico de Química General.
6. No omitas pasos de cálculo.
7. Si falta un dato necesario, indícalo claramente.

ESTRUCTURA OBLIGATORIA PARA EJERCICIOS:
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

INSTRUCCIÓN:
Resuelve el ejercicio usando la estructura obligatoria de 8 apartados.
Usa únicamente los datos de la pregunta real del estudiante.
El contexto recuperado sirve solo como referencia teórica.
""".strip()

        max_tokens = 2800
        temperature = 0.05

    respuesta = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_final},
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return respuesta.choices[0].message.content


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
    print("Modelo esperado: phi4_quimica_v2_BF16.gguf mediante llama-server")
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
