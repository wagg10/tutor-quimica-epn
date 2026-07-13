Tabla X. Resumen del entrenamiento del modelo Qwen2.5-14B-Instruct mediante QLoRA

| Indicador | Valor obtenido |
|---|---|
| Modelo base | Qwen/Qwen2.5-14B-Instruct |
| Método de adaptación | QLoRA 4-bit + PEFT |
| Dataset utilizado | dataset_quimica_final.jsonl |
| Número de épocas | 4 |
| Longitud máxima de secuencia | 3072 tokens |
| Batch efectivo | 8 |
| Steps por época | 76 |
| Total de steps planificados | 304 |
| Warmup steps | 15 |
| Tasa de aprendizaje | 0,0001 |
| Rango LoRA (r) | 64 |
| Alpha LoRA | 128 |
| Dropout LoRA | 0,1 |
| Pérdida de entrenamiento | 0,3688 |
| Pérdida de validación final | 0,4232 |
| Precisión token a token final | 88,70 % |
| Duración total del entrenamiento | 46,53 minutos |
| Step final alcanzado | 304 |
| Mejor checkpoint | checkpoint-152 |
| Mejor métrica registrada | 0,4232 |
| GPU utilizada | NVIDIA GeForce RTX 5090 |
| VRAM total disponible | 33,66 GB |
| VRAM libre inicial | 32,58 GB |
| Parámetros totales del modelo | 8.439.256.064 |
| Parámetros entrenables | 275.251.200 |
| Porcentaje de parámetros entrenables | 3,2616 % |
