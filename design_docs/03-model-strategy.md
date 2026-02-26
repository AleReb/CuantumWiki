# 03 - Estrategia de Modelo (Tono)

## Objetivo del clasificador

Clasificar tono en etiquetas iniciales:
- neutral
- positivo
- urgente
- molesto
- formal

## Restricciones ESP32-C3

- Priorizar modelos pequeños int8
- Input fijo (ej: 64 tokens)
- Parámetros recomendados: 10k a 120k

## Pipeline de entrenamiento

1. Curar dataset etiquetado
2. Entrenar baseline en PC
3. Cuantizar a int8 (TFLite Micro compatible)
4. Validar exactitud vs latencia
5. Exportar artefacto por versión (`tone-v1`, `tone-v2`)

## Métricas mínimas

- Accuracy macro >= 80% (inicio)
- Latencia por nodo <= 80 ms
- Memoria estable sin resets
