# Anexo: Pruebas de inferencia del modelo Phi-4 Química v2 en bfloat16

Este anexo contiene la evidencia de pruebas realizadas al modelo Phi-4 Reasoning adaptado mediante QLoRA para tutoría académica en Química. La prueba fue ejecutada cargando el modelo base en bfloat16 y aplicando el adaptador entrenado `modelo_tutor_quimica_phi4_v2`.

Los archivos incluidos son:

- `probar_modelo_phi4_v2_bf16.py`: script utilizado para cargar el modelo y ejecutar inferencias.
- `pruebas_inferencia_phi4_v2_bf16.txt`: registro completo de preguntas y respuestas generadas por el modelo.
- `tabla_evaluacion_inferencia_bf16.md`: tabla resumen con resultados esperados, resultados generados y errores observados.
- `gpu_inferencia_bf16.txt`: evidencia del uso de GPU durante la prueba.
- `versiones_inferencia_bf16.txt`: versiones de librerías empleadas para la inferencia.

Estas evidencias permiten analizar el comportamiento del modelo en ejercicios de conversión de temperatura y cálculo de masa molar, verificando tanto la estructura pedagógica de respuesta como la exactitud aritmética.
