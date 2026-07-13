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
        n_resultados=5
    )

    print("Fragmentos recuperados:")
    for item in resumen_recuperacion:
        print(item)
    print()

    prompt_usuario = f"""
Pregunta del estudiante:
{pregunta}

Contexto recuperado desde la base semántica del libro de Química:
{contexto}

INSTRUCCIONES DE USO DEL CONTEXTO:
1. Responde principalmente la pregunta del estudiante.
2. Usa el contexto recuperado como apoyo teórico.
3. No copies literalmente todo el contexto; intégralo de forma pedagógica.
4. Si el contexto recuperado no es suficiente, puedes complementar con conocimiento general de Química.
5. Si hay cálculos químicos, presenta: datos, fórmula, sustitución, operación y resultado.
6. No inventes páginas, capítulos ni fuentes que no estén en el contexto.
7. Mantén un estilo de tutor académico de Química General.
8. Evita responder sobre temas fuera de Química General.
""".strip()

    respuesta = client_llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": skill_pedagogica},
            {"role": "user", "content": prompt_usuario}
        ],
        temperature=0.15,
        max_tokens=3000
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
