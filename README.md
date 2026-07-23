# Tutor Académico de Fundamentos de Química con LLM — EPN

Tutor académico local desarrollado como parte del Trabajo de Integración Curricular de la carrera de Ingeniería de Software de la Escuela Politécnica Nacional.

El proyecto implementa un tutor para Fundamentos de Química capaz de:

- Responder preguntas conceptuales de Química General.
- Resolver ejercicios paso a paso.
- Recuperar información mediante una arquitectura RAG.
- Ejecutar modelos de lenguaje ajustados mediante QLoRA.
- Mostrar las respuestas mediante una interfaz desarrollada con Streamlit.

---

## Características

- Tutor especializado en contenidos de Fundamentos de Química.
- Resolución estructurada de ejercicios.
- Explicaciones conceptuales organizadas.
- Recuperación semántica mediante ChromaDB.
- Integración con modelos ejecutados localmente en LM Studio.
- Interfaz web desarrollada con Streamlit.
- Clasificación de consultas dentro y fuera del dominio.
- Uso de una guía pedagógica para organizar las respuestas.
- Procesamiento y normalización de expresiones matemáticas.

---

## Tecnologías utilizadas

- Python 3.12
- Streamlit
- LM Studio
- ChromaDB
- Hugging Face Transformers
- PyTorch
- PEFT
- TRL
- BitsAndBytes
- QLoRA
- GGUF
- CUDA
- Modelos de embeddings

### Modelos evaluados

- Phi-4-reasoning 14B
- Qwen2.5-14B-Instruct
- DeepSeek-R1-Distill-Qwen-14B

El modelo seleccionado para el prototipo fue **Phi-4-reasoning 14B**, ajustado mediante QLoRA y convertido al formato GGUF para su ejecución local.

---

## Requisitos

Para ejecutar el proyecto se recomienda disponer de:

- Ubuntu o una distribución Linux compatible.
- GPU NVIDIA.
- Controladores NVIDIA y CUDA correctamente instalados.
- Python 3.12.
- Entorno virtual del proyecto.
- LM Studio.
- Modelo ajustado en formato GGUF.
- Base semántica de ChromaDB.

Para verificar que la GPU sea reconocida:

```bash
nvidia-smi
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd Quimica
```

Reemplaza `URL_DEL_REPOSITORIO` por la dirección real del repositorio.

### 2. Crear el entorno virtual

```bash
python3 -m venv entorno_quimica_qlora
```

### 3. Activar el entorno virtual

```bash
source entorno_quimica_qlora/bin/activate
```

### 4. Instalar las dependencias

```bash
pip install -r requirements.txt
```

---

## Configuración de LM Studio

El sistema utiliza LM Studio para ejecutar el modelo localmente.

1. Abrir LM Studio.
2. Cargar el modelo del tutor en formato GGUF.
3. Asignar la mayor cantidad posible de capas a la GPU mediante **GPU Offload**.
4. Iniciar el servidor local.
5. Configurar el puerto `1234`.

Para comprobar que el servidor está funcionando:

```bash
curl http://127.0.0.1:1234/v1/models
```

La respuesta debe mostrar un JSON con el identificador del modelo cargado.

---

## Ejecución

El proyecto necesita dos componentes activos:

1. El servidor local de LM Studio.
2. La interfaz Streamlit.

### 1. Ejecutar LM Studio

Desde una terminal:

```bash
/home/lasinac/Descargas/squashfs-root/lm-studio --no-sandbox
```

Luego, cargar el modelo e iniciar el servidor local en el puerto `1234`.

### 2. Ejecutar la interfaz RAG

En una segunda terminal:

```bash
cd /home/lasinac/Quimica
source entorno_quimica_qlora/bin/activate
python -m streamlit run rag_quimica_general/interfaz_tutor_quimica.py
```

Para ejecutar la aplicación y guardar un respaldo de la salida:

```bash
cd /home/lasinac/Quimica
source entorno_quimica_qlora/bin/activate
python -m streamlit run rag_quimica_general/interfaz_tutor_quimica.py 2>&1 | tee respaldo_rag_anexo_vii.txt
```

Streamlit mostrará en la terminal la dirección local de la aplicación, normalmente:

```text
http://localhost:8501
```

La primera consulta puede tardar más debido a la carga del modelo de embeddings y de la base semántica.

---

## Configuración del proyecto

Los principales parámetros se encuentran al inicio del archivo:

```text
rag_quimica_general/interfaz_tutor_quimica.py
```

| Parámetro | Descripción |
|---|---|
| `LLM_BASE_URL` | Dirección del servidor local de LM Studio. |
| `LLM_MODEL` | Identificador del modelo cargado en LM Studio. |
| `DB_PATH` | Ruta de la base semántica de ChromaDB. |
| `COLLECTION_NAME` | Nombre de la colección utilizada. |
| `EMBEDDING_MODEL_NAME` | Modelo de embeddings para la recuperación semántica. |
| `N_FRAGMENTOS` | Cantidad de fragmentos recuperados por consulta. |
| `UMBRAL_DISTANCIA_QUIMICA` | Umbral utilizado para determinar si una consulta pertenece al dominio de Química. |

La dirección predeterminada del servidor es:

```text
http://127.0.0.1:1234/v1
```

Si LM Studio utiliza otro puerto o identificador de modelo, se deben actualizar `LLM_BASE_URL` y `LLM_MODEL`.

---

## Funcionamiento del sistema

El procesamiento de una consulta se realiza en cuatro etapas:

1. **Enrutamiento:** clasifica la entrada como saludo, consulta de Química o consulta fuera del dominio.
2. **Recuperación semántica:** busca en ChromaDB los fragmentos más relacionados con la pregunta.
3. **Generación:** el modelo produce una explicación conceptual o una resolución paso a paso.
4. **Postprocesamiento:** normaliza el formato matemático y limpia la respuesta antes de mostrarla.

Los saludos y las consultas que no pertenecen a Química se procesan sin activar la recuperación semántica.

---

## Estructura del proyecto

```text
Quimica/
├── README.md
├── requirements.txt
├── dataset_quimica_final.jsonl
├── modelo_tutor_quimica_phi4_v2/
├── checkpoints_phi4_v2/
├── checkpoints_qwen25_14b_v2_reentrenado/
├── checkpoints_deepseek_r1_qwen14b_v2/
├── scripts_entrenamiento/
├── trainer_states/
├── logs/
├── fusion_gguf/
├── skill/
├── evidencias/
├── rag/
├── rag_quimica_general/
├── evaluacion_independiente/
├── interfaz/
├── entorno/
└── muestra_corpus.jsonl
```

### Correspondencia con los anexos

| Carpeta o archivo | Contenido |
|---|---|
| `scripts_entrenamiento/` | Scripts de entrenamiento QLoRA de los modelos. |
| `trainer_states/` | Estados y métricas del entrenamiento. |
| `logs/` | Registros de entrenamiento y reentrenamiento. |
| `fusion_gguf/` | Fusión del adaptador y conversión del modelo a GGUF. |
| `skill/` | Guía pedagógica empleada por el tutor. |
| `evidencias/` | Evidencias de validación de la guía pedagógica. |
| `rag/` | Recursos de recuperación semántica. |
| `rag_quimica_general/` | Interfaz y lógica principal del tutor. |
| `evaluacion_independiente/` | Evaluación realizada con ejercicios externos. |
| `interfaz/` | Componentes de la aplicación Streamlit. |
| `entorno/` | Información sobre dependencias y verificación de GPU. |
| `muestra_corpus.jsonl` | Muestra del corpus conversacional. |

---

## Resultados principales

El sistema fue evaluado mediante 40 ejercicios externos al conjunto utilizado para el ajuste.

El modelo Phi-4-reasoning 14B obtuvo:

- **96,3 % de aciertos numéricos.**
- **90 % de cumplimiento de la estructura de respuesta esperada.**

Los principales errores identificados estuvieron relacionados con cálculos puntuales, traslado de valores intermedios y redondeo.

---

## Datos y modelos

Por restricciones de tamaño y derechos de autor, el repositorio no publica:

- Pesos completos de los modelos.
- Puntos de control de entrenamiento.
- Corpus institucional completo.
- Libro utilizado para construir la base semántica.
- Base completa de ChromaDB.

El repositorio incluye únicamente una muestra del corpus y los archivos necesarios para documentar el proceso realizado.

---

## Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| `Connection error` | El servidor de LM Studio no está activo. | Iniciar el servidor y verificarlo con `curl`. |
| Respuestas demasiado lentas | El modelo se está ejecutando en CPU. | Aumentar el valor de GPU Offload y comprobar el uso con `nvidia-smi`. |
| No se encuentra la guía pedagógica | La ruta configurada es incorrecta. | Revisar la variable `SKILL_PATHS`. |
| Error al cargar ChromaDB | La ruta o colección no coincide. | Revisar `DB_PATH` y `COLLECTION_NAME`. |
| El modelo no aparece | El valor de `LLM_MODEL` es incorrecto. | Copiar el identificador mostrado por LM Studio. |
| El puerto no responde | LM Studio utiliza otro puerto. | Actualizar `LLM_BASE_URL`. |
| Streamlit no inicia | El entorno no está activado o faltan dependencias. | Activar el entorno e instalar `requirements.txt`. |

---

## Autor

**Wilmer Guevara**  
Ingeniero de Software  
Escuela Politécnica Nacional

---

## Contexto académico

Trabajo de Integración Curricular  
Carrera de Ingeniería de Software  
Escuela Politécnica Nacional — EPN

---

## Licencia

Este proyecto tiene fines académicos.

Antes de reutilizar el código, los datos o los materiales incluidos, se deben revisar las restricciones institucionales y los derechos de autor correspondientes.
