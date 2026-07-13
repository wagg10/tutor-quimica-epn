# Evaluación cualitativa de inferencia - Phi-4 Química v2 en bfloat16

| N.º | Ejercicio | Resultado esperado | Resultado del modelo | Error | Evaluación |
|---:|---|---:|---:|---:|---|
| 1 | Hidrógeno líquido: 20,27 K → °F | -423,18 °F | -423,18 °F | 0,00 °F | Correcto |
| 2 | Nitrógeno líquido: -195,79 °C → K | 77,36 K | 77,36 K | 0,00 K | Correcto |
| 3 | Nitrógeno líquido: -195,79 °C → °F | -320,42 °F | -320,62 °F | 0,20 °F | Parcialmente correcto |
| 4 | Masa molar del H2SO4 | 98,08 g/mol | 98,08 g/mol | 0,00 g/mol | Correcto |

## Observación técnica

La inferencia en bfloat16 mejoró la precisión aritmética respecto a la prueba previa en 4-bit. El modelo resolvió correctamente la conversión de Kelvin a Fahrenheit para el hidrógeno líquido, la conversión de Celsius a Kelvin para el nitrógeno líquido y el cálculo de masa molar del ácido sulfúrico. Sin embargo, en la conversión de Celsius a Fahrenheit para el nitrógeno líquido se observó un error decimal de 0,20 °F, por lo que se recomienda mantener una etapa de verificación aritmética para ejercicios numéricos.

## Conclusión de la prueba

El modelo Phi-4 Química v2 cargado en bfloat16 evidencia una respuesta pedagógica estructurada y una mejora en la exactitud numérica. La configuración resulta adecuada para pruebas de máxima precisión, aunque para integración en Hermes se recomienda evaluar una cuantización Q8_0 como equilibrio entre precisión y consumo de memoria.
