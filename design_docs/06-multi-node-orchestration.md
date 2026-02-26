# 06 - Orquestación Multi-Nodo

## Estrategia inicial

- `route=auto` con round-robin entre 4 C3
- Excluir nodos con timeout consecutivo
- Reintegrar nodos tras N pings exitosos

## Modo resiliente

- Retry una vez en nodo alternativo
- Circuit breaker por nodo

## Escalado

- Soportar nuevos workers con registro dinámico
- Tabla de capacidades por nodo (modelo soportado, versión)
