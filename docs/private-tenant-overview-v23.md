# Churn Lab v2.3 — resumen agregado por empresa en staging privado

**Solo laboratorio local con cuentas y organizaciones ficticias.** No habilitar en Render ni introducir datos de clientes reales.

## Mejoras

- Al seleccionar una organización activa, el panel privado `/saas/v3/console` presenta cuatro indicadores derivados de los **últimos 20 análisis sintéticos disponibles**: análisis ejecutados, suma de perfiles evaluados, conteo agregado de alto riesgo y proporción agregada de alto riesgo.
- La proporción divide la suma de conteos de alto riesgo entre el total de perfiles evaluados en las ejecuciones mostradas; **no equivale al porcentaje de clientes reales**, ni al porcentaje de clientes únicos. Ejecutar los mismos tres ejemplos varias veces suma perfiles repetidos.
- El historial muestra también la proporción de alto riesgo en cada ejecución. Las cifras provienen de la respuesta autenticada del tenant activo: el navegador no guarda datos de terceros, CSV, identificadores ni predicciones individuales.
- Al cambiar de organización, las cifras se reinician inmediatamente. Las respuestas de historial o miembros del tenant anterior se descartan si llegan tarde. Las acciones del listado de miembros conservan el tenant de su fila; el servidor sigue autorizando cada operación.
- Pantalla adaptable a escritorio y móvil, con indicadores legibles y estados vacíos.

## Garantías y límites

- **Sin nuevos endpoints, migraciones o privilegios**: reutiliza los controles OIDC, la pertenencia verificada, RLS y el API de historial de v2.2.
- Los datos agregados se mantienen en los límites de la empresa; la página y los recursos siguen protegidos por sesión y banderas de staging.
- No se han implementado nombres de usuario provenientes de Keycloak, perfiles reales ni subida de ficheros. Esos cambios requieren un diseño de identidad y privacidad independiente.
- No modificar `render.yaml`, `Dockerfile.demo` ni las variables de la demo pública.

## Pruebas

- `PYTHONPATH=src pytest -q tests/test_staging_console.py` valida las rutas privadas y la inclusión de componentes con sesión de prueba.
- `PYTHONPATH=src pytest -q tests/test_staging_browser_e2e.py` con PostgreSQL efímero y Playwright en CI comprueba las cifras sobre historial totalmente ficticio y que no pasan de una empresa a otra.
- La CI existente ejecuta comprobaciones de sintaxis JavaScript y pruebas Python; el workflow opcional de SaaS incluye PostgreSQL 16, roles y navegador real.

Para actualizar un entorno privado local **ya creado**, después del merge ejecutar desde su directorio de proyecto:

```powershell
git pull --ff-only
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d --build app
```

Nunca usar `docker compose down --volumes`: elimina usuarios y datos ficticios de PostgreSQL/Keycloak y la CA local de Caddy.
