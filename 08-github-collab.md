# 08 - GitHub para trabajo multi-PC

## Recomendación

Crear repo `CuantumWiki` y sincronizar desde todos los PCs.

## Comandos base

```bash
cd C:\CuantumWiki
git init
git add .
git commit -m "init: arquitectura CuantumWiki"
git branch -M main
git remote add origin <URL_DEL_REPO>
git push -u origin main
```

## Flujo diario

```bash
git pull --rebase
git add .
git commit -m "update: ..."
git push
```

## Convención sugerida de ramas

- `main`: estable
- `dev`: integración
- `feat/*`: features puntuales
