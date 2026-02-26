# 02 - Protocolo Serial (Web <-> S3)

## Formato recomendado

JSON Lines (`\n` por mensaje).

## Mensajes Web -> S3

```json
{"type":"infer","req_id":"r-001","model":"tone-v1","tokens":[12,45,77],"max_len":64,"route":"auto"}
```

Campos:
- `req_id`: id único de solicitud
- `tokens`: input_ids ya tokenizados/padded
- `max_len`: longitud fija de entrada
- `route`: `auto | node_id`

## Mensajes S3 -> Web

```json
{"type":"result","req_id":"r-001","label":"neutral","confidence":0.91,"node":"c3-2","latency_ms":38}
```

Errores:

```json
{"type":"error","req_id":"r-001","code":"NODE_TIMEOUT","message":"worker no respondió"}
```

## Reglas

- Timeout inferencia por nodo: 300-800 ms
- Reintento: 1
- Si falla nodo asignado: reroute automático
