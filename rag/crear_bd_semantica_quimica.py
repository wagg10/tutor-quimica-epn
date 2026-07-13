import json
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


DB_PATH = "bd_semantica_quimica"
COLLECTION_NAME = "coleccion_libro_quimica_general"
REGISTRO_PATH = Path("registro_chunks_quimica.json")
LIBRO_DIR = Path("libro_quimica_general")


embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def limpiar_texto(texto: str) -> str:
    texto = texto.replace("\x00", " ")
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def obtener_numero_capitulo(nombre_archivo: str) -> int:
    """
    Intenta detectar el número de capítulo desde nombres como:
    capitulo_01_estructura_atomica.pdf
    capitulo_2_tabla_periodica.pdf
    """
    match = re.search(r"capitulo[_\-\s]*(\d+)", nombre_archivo.lower())
    if match:
        return int(match.group(1))
    return 0


def obtener_titulo_capitulo(path: Path) -> str:
    nombre = path.stem
    nombre = re.sub(r"capitulo[_\-\s]*\d+[_\-\s]*", "", nombre, flags=re.IGNORECASE)
    nombre = nombre.replace("_", " ").replace("-", " ")
    nombre = re.sub(r"\s+", " ", nombre).strip()
    return nombre.title() if nombre else path.stem


def extraer_paginas_pdf(path: Path):
    """
    Extrae el texto de cada página del PDF.
    Requiere: pip install pymupdf
    """
    import fitz

    paginas = []
    documento = fitz.open(path)

    for indice_pagina, pagina in enumerate(documento, start=1):
        texto = pagina.get_text("text")
        texto = limpiar_texto(texto)

        if texto:
            paginas.append({
                "pagina": indice_pagina,
                "texto": texto
            })

    documento.close()
    return paginas


def extraer_texto_txt_md(path: Path):
    texto = path.read_text(encoding="utf-8", errors="ignore")
    texto = limpiar_texto(texto)

    return [{
        "pagina": None,
        "texto": texto
    }]


def dividir_en_chunks(texto: str, tamano_chunk: int = 1300, solapamiento: int = 200):
    """
    Divide el texto en fragmentos con solapamiento.
    Esto ayuda a que las ideas no se corten bruscamente.
    """
    texto = limpiar_texto(texto)

    if len(texto) <= tamano_chunk:
        return [texto]

    chunks = []
    inicio = 0

    while inicio < len(texto):
        fin = inicio + tamano_chunk
        chunk = texto[inicio:fin].strip()

        if chunk:
            chunks.append(chunk)

        inicio = fin - solapamiento

        if inicio < 0:
            inicio = 0

        if inicio >= len(texto):
            break

    return chunks


def cargar_capitulos():
    archivos = sorted([
        path for path in LIBRO_DIR.iterdir()
        if path.suffix.lower() in [".pdf", ".txt", ".md"]
    ])

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos PDF, TXT o MD en la carpeta {LIBRO_DIR}"
        )

    documentos = []

    for path in archivos:
        capitulo = obtener_numero_capitulo(path.name)
        titulo_capitulo = obtener_titulo_capitulo(path)

        print(f"Procesando: {path.name}")

        if path.suffix.lower() == ".pdf":
            paginas = extraer_paginas_pdf(path)
        else:
            paginas = extraer_texto_txt_md(path)

        for pagina_item in paginas:
            pagina = pagina_item["pagina"]
            texto_pagina = pagina_item["texto"]

            chunks = dividir_en_chunks(texto_pagina)

            for indice_chunk, chunk in enumerate(chunks, start=1):
                documentos.append({
                    "capitulo": capitulo,
                    "titulo_capitulo": titulo_capitulo,
                    "fuente": path.name,
                    "pagina": pagina,
                    "indice_chunk": indice_chunk,
                    "contenido": chunk
                })

    return documentos


def crear_bd():
    documentos = cargar_capitulos()

    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Colección anterior eliminada: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={
            "descripcion": "Base semántica del libro de Química General por capítulos"
        }
    )

    ids = []
    texts = []
    metadatas = []
    registro = {}

    for i, doc in enumerate(documentos, start=1):
        capitulo = doc["capitulo"]
        pagina = doc["pagina"] if doc["pagina"] is not None else 0
        indice_chunk = doc["indice_chunk"]

        chunk_id = f"quimica_cap{capitulo:02d}_pag{pagina:04d}_chunk{indice_chunk:04d}_{i:06d}"

        texto_embedding = f"""
Capítulo {capitulo}: {doc["titulo_capitulo"]}
Página: {pagina if pagina else "No aplica"}

Contenido:
{doc["contenido"]}
""".strip()

        metadata = {
            "capitulo": capitulo,
            "titulo_capitulo": doc["titulo_capitulo"],
            "fuente": doc["fuente"],
            "pagina": pagina,
            "indice_chunk": indice_chunk,
            "tipo": "libro_quimica_general"
        }

        ids.append(chunk_id)
        texts.append(texto_embedding)
        metadatas.append(metadata)

        registro[chunk_id] = {
            "capitulo": capitulo,
            "titulo_capitulo": doc["titulo_capitulo"],
            "fuente": doc["fuente"],
            "pagina": pagina,
            "indice_chunk": indice_chunk,
            "contenido": doc["contenido"]
        }

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )

    with open(REGISTRO_PATH, "w", encoding="utf-8") as f:
        json.dump(registro, f, ensure_ascii=False, indent=2)

    print("\nBase semántica creada correctamente.")
    print(f"Base ChromaDB: {DB_PATH}")
    print(f"Colección: {COLLECTION_NAME}")
    print(f"Total de chunks guardados: {len(ids)}")
    print(f"Registro JSON: {REGISTRO_PATH}")


if __name__ == "__main__":
    crear_bd()
