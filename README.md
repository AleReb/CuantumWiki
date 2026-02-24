# CuantumWiki

Wiki técnico para el proyecto de detección de tono con arquitectura distribuida:
- Web (tokenización + UI)
- ESP32-S3 (coordinador por serial)
- 4x ESP32-C3 (workers TinyML)

## Estructura

- `01-system-architecture.md`
- `02-serial-protocol.md`
- `03-model-strategy.md`
- `04-firmware-plan.md`
- `05-web-tokenizer-plan.md`
- `06-multi-node-orchestration.md`
- `07-roadmap.md`
- `08-github-collab.md`

## Objetivo

Detectar tono/intención de mensajes de texto con baja latencia, usando tokenización en PC y clasificación liviana en nodos ESP32.
