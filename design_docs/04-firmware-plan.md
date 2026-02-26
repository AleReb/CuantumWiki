# 04 - Plan de Firmware

## ESP32-S3 (Coordinator)

Responsabilidades:
- Parser de mensajes serial JSONL
- Cola de requests
- Scheduler de workers
- Monitor de salud (`ping`, `last_seen`, `error_rate`)

## ESP32-C3 (Worker)

Responsabilidades:
- Inicializar modelo TinyML
- Recibir features/tokens
- Ejecutar inferencia
- Responder `label/confidence/latency`

## Versionado

- API firmware: `fw_api_v1`
- Modelo: `tone-vX`

## Logging

- S3: logs de enrutamiento + errores
- C3: logs compactos (boot, infer ok/fail)
