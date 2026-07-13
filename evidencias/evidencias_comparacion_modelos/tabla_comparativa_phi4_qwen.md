# Comparación cualitativa entre Phi-4 Reasoning y Qwen2.5-14B-Instruct

| Criterio | Phi-4 Reasoning v2 | Qwen2.5-14B-Instruct v2 |
|---|---|---|
| Método de adaptación | QLoRA 4-bit + PEFT | QLoRA 4-bit + PEFT |
| Dataset utilizado | dataset_quimica_final.jsonl | dataset_quimica_final.jsonl |
| Formato pedagógico de ocho secciones | Correcto | Correcto |
| Inferencia en bfloat16 | Funcional en la RTX 5090 | Muy pesada; produjo descarga parcial a CPU |
| Inferencia práctica | Estable en bfloat16 | Funcional en 8-bit |
| Conversión 20,27 K → °F | Correcta en bfloat16 | Correcta en 8-bit |
| Conversión -195,79 °C → K | Correcta | Correcta |
| Conversión -195,79 °C → °F | Error decimal menor en una prueba | Error fuerte por arrastre de dato |
| Masa molar H2SO4 | Correcta | Correcta |
| Confiabilidad aritmética observada | Mayor | Parcial |
| Modelo seleccionado | Sí | No |

## Decisión

Phi-4 Reasoning v2 fue seleccionado como modelo principal debido a que presentó mayor estabilidad en inferencia, mejor manejo de los datos del enunciado y menor incidencia de errores aritméticos en las pruebas cualitativas. Qwen2.5-14B-Instruct v2 se mantuvo como modelo comparativo, ya que permitió contrastar el desempeño de otra arquitectura de 14B parámetros bajo el mismo dataset y método de adaptación.
