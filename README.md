# Tutor académico de Fundamentos de Química con LLM — EPN
Código y evidencias del Trabajo de Integración Curricular (Ingeniería de Software, EPN):
ajuste fino eficiente (QLoRA 4 bits + PEFT) de Phi-4-reasoning, Qwen2.5-14B-Instruct y
DeepSeek-R1-Distill-Qwen-14B para tutoría de Fundamentos de Química, con recuperación
semántica (RAG) y prototipo local en Streamlit.

## Estructura y correspondencia con los anexos del documento
- scripts_entrenamiento/ y trainer_states/ — entrenamiento QLoRA de los tres modelos (Anexo III; Tablas 9-10)
- logs/ — bitácoras de entrenamiento y reentrenamiento
- fusion_gguf/ — fusión del adaptador (checkpoint 152) y conversión a GGUF BF16 (Anexo IV)
- skill/ y evidencias/ — skill pedagógica y su validación (Anexo V)
- rag/ — base semántica en ChromaDB y tutor RAG (Anexo VI)
- evaluacion_independiente/ — evaluación sobre 40 ejercicios (Anexo VII)
- interfaz/ — aplicación Streamlit del prototipo (Anexo VIII)
- entorno/ — versiones del entorno (Tabla 6) y verificación de GPU
- muestra_corpus.jsonl — muestra del corpus conversacional (5 ejemplos)

## Nota sobre datos y modelos
Los pesos de los modelos, los puntos de control, el corpus completo y la base semántica
no se publican por tamaño y derechos de autor; se incluye una muestra del corpus.
