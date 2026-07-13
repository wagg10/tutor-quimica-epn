import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


DB_PATH = "bd_semantica_quimica"
COLLECTION_NAME = "coleccion_libro_quimica_general"
REGISTRO_PATH = Path("registro_chunks_quimica.json")


embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)

with open(REGISTRO_PATH, "r", encoding="utf-8") as f:
    registro = json.load(f)


print("Prueba de base semántica de Química")
print("Escribe 'salir' para terminar.\n")

while True:
    pregunta = input("Pregunta de prueba: ").strip()

    if pregunta.lower() in ["salir", "exit", "quit"]:
        print("Prueba finalizada.")
        break

    if not pregunta:
        continue

    resultados = collection.query(
        query_texts=[pregunta],
        n_results=5
    )

    print("\nResultados recuperados:")

    for pos, chunk_id in enumerate(resultados["ids"][0], start=1):
        distancia = resultados["distances"][0][pos - 1]
        item = registro[chunk_id]

        print("\n" + "=" * 80)
        print(f"Resultado {pos}")
        print(f"ID: {chunk_id}")
        print(f"Distancia semántica: {distancia:.4f}")
        print(f"Capítulo: {item['capitulo']}")
        print(f"Título capítulo: {item['titulo_capitulo']}")
        print(f"Fuente: {item['fuente']}")
        print(f"Página: {item['pagina']}")

        print("\nContenido recuperado:")
        print(item["contenido"][:1200])

    print("\n" + "=" * 100 + "\n")
