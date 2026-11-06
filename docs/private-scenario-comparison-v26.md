# Churn Lab v2.6 — comparación simultánea de escenarios sintéticos

**Exclusivamente para staging privado local.** No introducir perfiles de clientes reales ni exponer rutas privadas en la demo pública.

## Qué permite

La consola `/saas/v3/console` añade un botón **«Comparar y guardar los 3 escenarios»**. Al pulsarlo, se puntúan tres grupos distintos de perfiles **ficticios y definidos en el servidor**, usando el mismo modelo entrenado:

- **Mixto:** los tres perfiles ficticios originales de v2.2.
- **Riesgo potencial:** tres casos inventados con contratos mensuales y antigüedad reducida.
- **Fidelización:** tres casos inventados con contratos de mayor duración.

La respuesta permite visualizar, para cada escenario, el número de perfiles ficticios puntuados, cuántos quedan por encima del umbral de clasificación y su probabilidad media. Se presenta en tres tarjetas lado a lado en escritorio (apiladas en pantallas estrechas).

**Interpretación:** no son cohortes de clientes reales, ni predicciones de tendencias temporales, ni resultados de un ensayo. Son experimentos independientes con ejemplos distintos. No sirven para afirmar que un escenario causa una subida o bajada real del churn.

## Seguridad y conservación de datos

El endpoint nuevo es `POST /saas/v3/organizations/{tenant_id}/runs/compare`, disponible **solo** con autenticación y para roles `owner` o `analyst` de la empresa. Un `viewer` puede leer el historial de su empresa, pero no crear análisis. Se exige origen HTTPS de confianza y cuerpo JSON vacío `{}`; los campos inesperados, clientes enviados desde el navegador, CSV y escenarios arbitrarios son rechazados.

Los tres resultados se ejecutan en una **misma transacción PostgreSQL**: se insertan tres registros de historial con **solo** empresa, escenario, contadores agregados, autor técnico ya existente y fecha. Si falla una predicción, se revierte la transacción completa y no quedan registros parcialmente guardados. No se almacenan perfiles ni predicciones individuales.

El historial conserva el límite actual: **20 ejecuciones recientes por empresa**. Cada comparación consume **tres** entradas; las ejecuciones más antiguas pueden dejar de mostrarse al sobrepasar 20, pero no se borran de la base de datos. El panel continúa protegido por OIDC y aislamiento de empresas (PostgreSQL RLS).

## Despliegue

**No hay migración SQL nueva en v2.6.** Reutiliza el esquema `0005_synthetic_scenarios.sql`, que ya se aplicó al instalar v2.5. Esta versión NO modifica `render.yaml`, `Dockerfile.demo`, variables de Render ni ninguna ruta de la demo pública.

Tras fusionar el PR, en tu PowerShell local:

```powershell
cd C:\Users\arnau\proyectos\saas-churn-mlops
git pull --ff-only
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d --build app
docker compose --env-file .env.staging.local -f compose.private-staging.yml ps
```

Comprueba que `app`, `db`, `keycloak` y `proxy` estén activos y PostgreSQL esté `Healthy`. Si caduca la sesión, inicia sesión de nuevo en `https://app.localhost/saas/v3/login`. Luego abre `https://app.localhost/saas/v3/console` y comprueba que **Acme Demo** y su historial continúen intactos.

**No repitas las migraciones 0004 ni 0005; no ejecutes `git stash pop` ni `docker compose down --volumes`.**

## Pruebas previstas

- `tests/test_staging_analysis.py`: autorización por rol, rechazo de entradas ajenas a los tres escenarios, aislamiento por organización, respuesta agregada y reversión atómica cuando falla la segunda predicción.
- `tests/test_staging_console.py`: rutas privadas y presencia de componentes sin inyección de HTML ni almacenamiento local.
- `tests/test_staging_browser_e2e.py`: comparación en Chromium, tres tarjetas de resultados, solicitud vacía `{}`, limpieza de datos al cambiar de empresa y botón deshabilitado para espectadores.
- CI normal: tests de Python, sintaxis JavaScript, inferencia y construcción de Docker. Seguridad SaaS: PostgreSQL 16 efímero con RLS y navegador real. QA: escritorio y móvil.
