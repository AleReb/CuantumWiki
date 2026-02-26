# 01 - Arquitectura del Sistema

## Componentes

1. **Web App (PC)**
   - Entrada de texto
   - Tokenización
   - Preprocesamiento y empaquetado
   - Visualización de resultados

2. **ESP32-S3 (Coordinator)**
   - Conexión Serial con la Web
   - Enrutamiento de tareas a nodos C3
   - Agregación de resultados
   - Health checks de nodos

3. **4x ESP32-C3 (Workers)**
   - Inferencia TinyML (modelo int8)
   - Respuesta con clase + confianza + latencia

## Flujo

Web -> Serial -> S3 -> C3[n] -> S3 -> Web

## Principios

- El texto crudo se procesa en PC.
- A microcontroladores se envían solo tensores/features compactas.
- El S3 desacopla frontend del clúster de workers.
