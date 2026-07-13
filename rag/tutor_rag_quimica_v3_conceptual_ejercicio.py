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

    prompt_usuario = f"""
PREGUNTA REAL DEL ESTUDIANTE:
{pregunta}

FRAGMENTOS RECUPERADOS DESDE EL LIBRO DE QUÍMICA:
{contexto}

REGLAS OBLIGATORIAS:
1. Responde únicamente la pregunta real del estudiante.
2. No inventes elementos químicos, compuestos, datos numéricos ni ejemplos que el estudiante no haya pedido.
3. Si la pregunta es conceptual, responde como explicación teórica breve.
4. Si la pregunta empieza con "qué es", "define", "explica", "diferencia" o "cómo se calcula", NO uses formato de ejercicio largo.
5. Para preguntas conceptuales usa esta estructura:
   - Definición
   - Explicación
   - Importancia en Química
   - Ejemplo solo si el estudiante lo pide
6. Para ejercicios numéricos usa esta estructura:
   - Datos
   - Fórmula
   - Sustitución
   - Operación
   - Resultado
   - Conclusión
7. El contexto recuperado es apoyo teórico, no es el enunciado del problema.
8. No copies literalmente todo el contexto.
9. No menciones capítulos o páginas a menos que el estudiante lo pida.
10. Mantén un estilo de tutor académico de Química General.
""".strip()

    system_final = f"""
{skill_pedagogica}

REGLA SUPERIOR PARA RAG:
El contexto recuperado NO reemplaza la pregunta del estudiante.
Nunca cambies la pregunta original.
Nunca conviertas una pregunta conceptual en un ejercicio.
No agregues ejemplos específicos si el estudiante no los solicita.
""".strip()

    respuesta = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_final},
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=0.03,
        max_tokens=1800
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
