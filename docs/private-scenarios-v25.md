# Churn Lab v2.5 — escenarios sintéticos seleccionables

**Entorno privado local exclusivamente.** Sin clientes reales, CSV, facturación ni acceso a datos personales.

## Qué añade

- Tres grupos de perfiles **inventados y preparados en el servidor**: `baseline` (mixto, los tres ejemplos originales de v2.2), `at_risk` (contratos mensuales con escasa antigüedad) y `loyal` (contratos largos y mayor antigüedad).
- Cada ejecución sigue calculando predicciones del modelo sobre **tres ejemplos ficticios**, pero el usuario puede escoger el escenario antes de iniciar el análisis.
- El servidor acepta **solo** `{"scenario": "baseline"|"at_risk"|"loyal"}`; el campo es opcional y por defecto es `baseline` para conservar compatibilidad. Pydantic rechaza claves adicionales, perfiles, CSV y escenarios no contemplados.
- La respuesta de ejecución y el historial incluyen el identificador público del escenario; **PostgreSQL solo guarda el nombre del escenario y los recuentos agregados**, nunca las filas de ejemplo, probabilidades por individuo ni detalles del modelo.
- En el gráfico e historial privados aparece la categoría del escenario. Si los dos análisis recientes son escenarios distintos, la UI **no presenta una subida/bajada como evolución temporal**: muestra una advertencia de que no se trata de un mismo grupo observado.

**Interpretación**: los perfiles no son clientes, los resultados no describen segmentos reales ni cambios en el churn observado. Repetir análisis del mismo escenario producirá normalmente resultados iguales para un modelo fijo. La v2.4 conserva el límite de **20 ejecuciones** por empresa en el historial visible.

## Seguridad

El endpoint autenticado reutiliza controles de sesión OIDC, origen HTTPS y pertenencia PostgreSQL con RLS, además de la autorización `owner/analyst`. `viewer` solo puede leer los agregados de su empresa. No se crean rutas públicas, permisos adicionales ni almacenamiento en el navegador.

El Dockerfile y `render.yaml` de la demo pública **no se modifican**.

## Migración 0005 — actualizar una instalación local existente

La versión 2.5 añade una columna `scenario_key` a `saas.sandbox_runs` con valor por defecto `baseline`. **Todos los análisis previos pasan a figurar como «Mixto»**, sin eliminar ni modificar sus conteos. Para bases de datos nuevas, el script local `initdb` aplica las migraciones 0001-0005 en orden.

En PostgreSQL **ya creado** (como tu instalación actual de Acme Demo), el contenedor NO ejecuta de nuevo `initdb`. Aplica **una sola vez** `db/migrations/0005_synthetic_scenarios.sql` antes de reconstruir `app`.

En PowerShell, desde el repositorio, **después del merge**:

```powershell
cd C:\Users\arnau\proyectos\saas-churn-mlops
git pull --ff-only
docker compose --env-file .env.staging.local -f compose.private-staging.yml ps
```

Antes de cualquier cambio de base de datos, crea una **copia nueva y verificada** (las copias de v2.3 son anteriores a tus análisis nuevos):

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml exec -T db pg_dump -U postgres -d churn_saas_staging -Fc -f /tmp/churn_backup_v25.dump
docker compose --env-file .env.staging.local -f compose.private-staging.yml cp db:/tmp/churn_backup_v25.dump ..\churn_backup_v25.dump
Get-Item ..\churn_backup_v25.dump | Select-Object Name, Length
```

Confirma que el archivo tenga tamaño superior a cero. Después aplica:

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml exec -T db psql -U postgres -d churn_saas_staging -v ON_ERROR_STOP=1 -f /churn-migrations/0005_synthetic_scenarios.sql
```

La salida correcta debe acabar en `COMMIT`. **No repitas esta migración**; si hay un error, consérvalo para diagnosticarlo antes de tocar Docker. Finalmente:

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d --build app
docker compose --env-file .env.staging.local -f compose.private-staging.yml ps
```

No ejecutes `git stash pop` ni `docker compose down --volumes`. Conserva los volúmenes de PostgreSQL, Keycloak y Caddy. Si la sesión caduca por la reconstrucción, vuelve a `https://app.localhost/saas/v3/login` y entra en `https://app.localhost/saas/v3/console`.

## Pruebas de GitHub Actions

El pipeline privado aplica la migración 0005 sobre PostgreSQL 16 efímero, prueba que los roles y tenants permanecen aislados y que solo se pueden ejecutar escenarios aceptados. Los tests de navegador verifican selección, presentación de escenarios y la advertencia al comparar dos grupos diferentes. La CI estándar comprueba la sintaxis JavaScript, entrenamiento y construcción de imágenes.
