# Churn Lab v2.3 — resumen agregado por empresa en staging privado

**Solo laboratorio local con cuentas y organizaciones ficticias.** No habilitar en Render ni introducir datos de clientes reales.

## Mejoras

- Al seleccionar una organización activa, el panel privado `/saas/v3/console` presenta cuatro indicadores derivados de los **últimos 20 análisis sintéticos disponibles**: análisis ejecutados, suma de perfiles evaluados, conteo agregado de alto riesgo y proporción agregada de alto riesgo.
- La proporción divide la suma de conteos de alto riesgo entre el total de perfiles evaluados en las ejecuciones mostradas; **no equivale al porcentaje de clientes reales**, ni al porcentaje de clientes únicos. Ejecutar los mismos tres ejemplos varias veces suma perfiles repetidos.
- El historial muestra también la proporción de alto riesgo en cada ejecución. Las cifras provienen de la respuesta autenticada del tenant activo: el navegador no guarda datos de terceros, CSV, identificadores ni predicciones individuales.
- Al cambiar de organización, las cifras se reinician inmediatamente. Las respuestas de historial o miembros del tenant anterior se descartan si llegan tarde. Las acciones del listado de miembros conservan el tenant de su fila; el servidor sigue autorizando cada operación.
- Pantalla adaptable a escritorio y móvil, con indicadores legibles y estados vacíos.

## Garantías y límites

- Las métricas reutilizan las rutas de historial de v2.2, sin nuevos privilegios de tabla. Los alias ficticios sí requieren una **migración nueva 0004**, un endpoint de edición propia y una función SQL autorizada únicamente para la identidad activa.
- Cada miembro puede asignarse un alias distinto de 2 a 40 caracteres **para cada empresa**. Los propietarios ven alias en el equipo en lugar del UUID completo; quien no haya elegido uno aparece como «Miembro sin alias» con un prefijo corto del identificador. El alias es voluntario, no se importa de Keycloak y no debe incluir datos reales.
- Los alias activos tienen unicidad sin distinguir mayúsculas/minúsculas dentro de cada empresa; la base de datos rechaza suplantaciones por duplicado.
- Los datos agregados se mantienen en los límites de la empresa; la página y los recursos siguen protegidos por sesión y banderas de staging.
- No se han implementado nombres de usuario provenientes de Keycloak, perfiles reales ni subida de ficheros. Esos cambios requieren un diseño de identidad y privacidad independiente.
- No modificar `render.yaml`, `Dockerfile.demo` ni las variables de la demo pública.

## Pruebas

- `PYTHONPATH=src pytest -q tests/test_staging_console.py` valida las rutas privadas y la inclusión de componentes con sesión de prueba.
- `PYTHONPATH=src pytest -q tests/test_staging_browser_e2e.py` con PostgreSQL efímero y Playwright en CI comprueba las cifras sobre historial totalmente ficticio y que no pasan de una empresa a otra.
- La CI existente ejecuta comprobaciones de sintaxis JavaScript y pruebas Python; el workflow opcional de SaaS incluye PostgreSQL 16, roles y navegador real.

### Actualización no destructiva de PostgreSQL existente

**IMPORTANTE:** Para entornos ya creados con la v2.2, los scripts de `docker-entrypoint-initdb.d` NO se vuelven a ejecutar porque el volumen contiene datos. Aplica la migración 0004 **una vez como superusuario local de migraciones**, después del merge/`git pull`, **antes** de iniciar la imagen nueva. Usa la misma instalación Compose:

```powershell
git pull --ff-only
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d db
docker compose --env-file .env.staging.local -f compose.private-staging.yml exec -T db psql -U postgres -d churn_saas_staging -v ON_ERROR_STOP=1 -f /churn-migrations/0004_member_aliases.sql
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d --build app
```

Esta migración añade una columna anulable y dos funciones SQL; **no borra** empresas, invitaciones, usuarios ni historial existente. **No la ejecutes dos veces**: es una migración de esquema versionada. Confirma previamente que estás en tu entorno de pruebas privado y realiza una copia de seguridad si necesitas conservarlo.

```powershell
git pull --ff-only
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d --build app
```

Nunca usar `docker compose down --volumes`: elimina usuarios y datos ficticios de PostgreSQL/Keycloak y la CA local de Caddy.
