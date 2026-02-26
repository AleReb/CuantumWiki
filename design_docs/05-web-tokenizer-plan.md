# 05 - Web + Tokenización

## Stack sugerido

- Frontend: React/Vite (o simple HTML+JS)
- Serial: Web Serial API
- Tokenización: librería JS (según vocab del modelo)

## Flujo UI

1. Usuario escribe mensaje
2. Web tokeniza y aplica padding
3. Envía `infer` por serial
4. Muestra resultado + confianza + nodo + latencia

## Buenas prácticas

- Cachear vocab/tokenizer en browser
- Validar longitud máxima de entrada
- Botón de reconnect serial
