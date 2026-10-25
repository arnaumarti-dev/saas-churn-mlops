# Churn Lab v2.4 — evolución y filtros de ejecuciones sintéticas

**Staging privado local, desactivado por defecto.** No incluir información real de clientes, facturación ni datos personales.

## Qué cambia

La página autenticada `/saas/v3/console` conserva los indicadores de v2.3 y añade:

- Filtro de **todos los registros disponibles**, **7 días** o **30 días**, aplicado al historial de la organización seleccionada.
- Gráfico de barras accesible con la **proporción de perfiles ficticios clasificados de alto riesgo en cada ejecución**, en orden cronológico.
- Comparación en **puntos porcentuales** entre los **dos últimos análisis del periodo seleccionado** (sube, baja o sin cambio).
- Indicadores y lista histórica que se actualizan al cambiar el filtro. Cambiar de empresa limpia los datos en pantalla y descarta solicitudes de red antiguas.
- Estados vacíos explícitos cuando no hay ejecuciones en el periodo o solo existe una para comparar.

### Interpretación correcta de los datos

El servidor devuelve exclusivamente los **20 análisis sintéticos más recientes** de la organización. Los filtros se aplican sobre **esos 20** registros; seleccionar 30 días **no consulta todo el historial de 30 días** si hay más de 20 ejecuciones. No hay paginación ni ampliación de datos en esta versión.

Cada análisis vuelve a procesar tres perfiles ficticios del mismo modelo. El porcentaje por ejecución y su diferencia no son indicadores de personas distintas, no prueban degradación del modelo ni representan cambios observados en la población de clientes reales. **No hay datos personales ni predicciones individuales persistidos**.

La interfaz usa HTML/CSS/JavaScript del mismo origen, sin CDN ni bibliotecas de gráficos externas. Solo renderiza recuentos y fechas devueltos por el endpoint protegido, con validación de rangos antes de calcular porcentajes y datos textuales escritos mediante `textContent`.

## Seguridad y compatibilidad

- **Sin rutas nuevas, sin migración nueva, sin cambios en permisos**: reutiliza `GET /saas/v3/organizations/{tenant_id}/runs` con la misma sesión OIDC y control de pertenencia PostgreSQL/RLS.
- El filtro se ejecuta **solo en memoria del navegador**: no hay almacenamiento local, cookies nuevas ni exportación de datos.
- La aplicación privada continúa disponible únicamente con sus flags de staging. **No modificar `render.yaml`, `Dockerfile.demo` ni el despliegue público de Render.**

## Pruebas

- `node --check src/churn_mlops/api/private_assets/console.js` y la CI JavaScript existente.
- `PYTHONPATH=src pytest -q tests/test_staging_console.py` confirma la autenticación y los controles nuevos.
- Las pruebas con Playwright y PostgreSQL efímero verifican proporciones, comparación, filtros 7/30 días, cambio de tenant y el historial por empresa.

## Actualizar una instalación privada ya preparada

**Solo después de fusionar el PR** y si tu staging de v2.3 está funcionando:

```powershell
cd C:\Users\arnau\proyectos\saas-churn-mlops
git pull --ff-only
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d --build app
docker compose --env-file .env.staging.local -f compose.private-staging.yml ps
```

En v2.4 **no hay migración SQL adicional**: **no repitas la migración 0004** que ya aplicaste para v2.3. Mantén los volúmenes persistentes de PostgreSQL, Keycloak y Caddy. **Nunca uses `docker compose down --volumes`** como parte de esta actualización.

Inicia sesión en `https://app.localhost/saas/v3/login`, entra en `https://app.localhost/saas/v3/console`, selecciona **Acme Demo** y cambia los filtros. Con una sola ejecución ya podrás ver una barra; la comparativa aparecerá tras la segunda ejecución ficticia.
