# Reporte resumido para tesis - Qwen2.5-14B-Instruct v2

| Indicador | Valor |
|---|---:|
| Modelo base | Qwen/Qwen2.5-14B-Instruct |
| Método | QLoRA 4-bit + PEFT |
| Dataset | dataset_quimica_final.jsonl |
| Épocas | 4 |
| Longitud máxima de secuencia | 3072 tokens |
| Batch efectivo | 8 |
| Steps por época | 76 |
| Total steps planificados | 304 |
| Warmup steps | 15 |
| Learning rate | 0.0001 |
| LoRA r | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.1 |
| Training loss | 0.3688 |
| Eval loss final | 0.4232 |
| Token accuracy final | 88.7 % |
| Duración total min | 46.53 |
| Global step final | 304 |
| Mejor checkpoint | ./checkpoints_qwen25_14b_v2/checkpoint-152 |
| Mejor métrica valor | 0.4232 |
| GPU | NVIDIA GeForce RTX 5090 |
| VRAM total GB | 33.66 |
| VRAM libre inicial GB | 32.58 |
| Parámetros totales | 8439256064 |
| Parámetros entrenables | 275251200 |
| Porcentaje entrenable | 3.2616 % |
