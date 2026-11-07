# Churn Lab v2.7 — contrato de datos y preparación para un piloto SaaS

**Estado:** comprobador de preparación **estructural** y demostración con datos inventados. **No es un modelo de churn SaaS validado, una ingesta comercial ni una certificación RGPD.** El modelo predictivo existente se entrenó con el conjunto público IBM Telco y permanece **separado** de esta nueva utilidad.

## El problema que resuelve

Antes de ofrecer una predicción de cancelaciones a una empresa SaaS hay que definir una etiqueta verificable, disponer de suficientes suscripciones y ventanas temporales históricas, y poder evaluar con un conjunto de prueba cronológicamente posterior al entrenamiento. Muchos prototipos confunden un dataset sintético con evidencia predictiva real.

La v2.7 prepara un **contrato de datos concreto** para discusiones con los primeros clientes y un **comprobador offline**, sin nuevos endpoints, sin subidas a la plataforma, sin base de datos añadida y sin almacenar filas en Churn Lab.

## Contrato inicial `saas-subscriptions-v1`

El fichero CSV debe contener **exactamente estas ocho columnas**, en cualquier orden:

| Campo | Significado | Validación |
| --- | --- | --- |
| `account_id` | Identificador opaco de una sola cuenta (seudónimo, no email ni nombre) | 6–64 letras/números/guiones |
| `snapshot_date` | Fecha histórica de corte donde se habrían conocido las variables | ISO `YYYY-MM-DD` |
| `tenure_months` | Antigüedad de la suscripción **hasta** el corte | Entero, 0–600 |
| `mrr_eur` | Ingreso mensual recurrente en EUR **en el corte** | 0–1.000.000, hasta dos decimales |
| `licensed_seats` | Número de licencias contratadas en el corte | Entero, 1–100.000 |
| `active_days_30d` | Días activos **anteriores** al corte, no futuros | Entero, 0–30 |
| `support_tickets_30d` | Incidencias abiertas durante los 30 días **anteriores** | Entero, 0–10.000 |
| `churn_next_30d` | **Etiqueta posterior:** la suscripción se cancela en los 30 días siguientes al corte | Solo 0 o 1 |

Es obligatorio definir con la empresa qué significa una cancelación (incluidas renovaciones fallidas, pausas o downgrade) **antes** de construir etiquetas. Una cuenta aparece como máximo una vez en este contrato inicial, para evitar mezclar instantáneas repetidas del mismo cliente en entrenamiento y pruebas. Si el cliente necesita series temporales por cuenta se diseñará más adelante un método de partición por entidad y tiempo.

No se admiten columnas `email`, teléfono, nombres ni datos libres; aun así, un token opaco no garantiza por sí solo anonimización legal. La herramienta **no puede comprobar** que las variables fueran realmente conocidas en la fecha de corte ni que el CSV provenga de un proceso legítimo; estos puntos requieren auditoría de origen y controles de privacidad externos.

## Comprobación offline reproducible

Desde la raíz del repositorio, con Python 3.11+:

```bash
PYTHONPATH=src python scripts/check_saas_pilot.py --demo --as-of 2026-03-01
```

En **PowerShell**:

```powershell
$env:PYTHONPATH="src"
python scripts/check_saas_pilot.py --demo --as-of 2026-03-01
```

Genera **240 cuentas y 12 fechas de corte, todas ficticias** y deterministas. No requiere servidor, red, Docker, usuario, contraseña ni librerías de entrenamiento adicionales.

El JSON generado contiene: número de filas y etiquetas positivas, periodos distintos, una partición **cronológica** entrenamiento/validación/prueba, número de eventos por partición, incidencias bloqueantes, fecha `as_of` y limitaciones explícitas. **No imprime identificadores o filas**. Se considera listo *solo para explorar un posible piloto*, no listo para comercializar un modelo.

Los criterios iniciales, **no estándares científicos validados**, son: al menos 180 cuentas independientes, seis fechas distintas, un mínimo de 30 cuentas y cinco cancelaciones por partición, y que cada etiqueta disponga de **30 días completos** posteriores al corte observados hasta `--as-of`. Una partición no debe mezclar distintas filas de la misma fecha de corte. Estos mínimos son orientativos y podrían ser insuficientes para entrenar un modelo estable; habrá que justificarlos estadísticamente para cada cliente y tasa de cancelación real.

### CSV local autorizado, solo cuando exista un proceso seguro

La utilidad también puede comprobar un CSV **local** seudonimizado sin subirlo a ChatGPT, Render ni a la consola privada. Requiere confirmar que el usuario tiene autorización:

```bash
PYTHONPATH=src python scripts/check_saas_pilot.py --csv /ruta/datos_anonimizados.csv --as-of 2026-03-01 --confirm-authorized-offline-data
```

**No uses esta opción con datos reales hasta disponer de autorización expresa, una base jurídica adecuada, documentación de protección de datos y un entorno local controlado.** La opción de confirmación no sustituye esos controles. El programa no envía datos por red, pero sí los lee de la máquina donde se ejecuta; no es un servicio de tratamiento de datos personales.

El CLI sale con código `0` si se cumplen las comprobaciones estructurales, `1` cuando existen obstáculos para el piloto y `2` ante un error de formato/lectura. Puede terminar sin mostrar entradas individuales ni rutas de archivo.

## Qué NO demuestra v2.7

- Que el antiguo modelo Telco funcione con datos SaaS: **no se reutiliza ni se reentrena aquí**.
- Que haya precisión, calibración o utilidad económica demostrada. No se obtienen AUC, F1 ni resultados de negocio.
- Que una intervención reduzca cancelaciones. Hacen falta hipótesis causales y pruebas apropiadas.
- Que el dataset esté libre de sesgos, fugas de futuro, relaciones personales o duplicados reales entre distintas cuentas.
- Que esté preparado para recibir datos personales por HTTP, facturar, incorporar clientes en producción o cumplir automáticamente el RGPD.

**Siguiente paso recomendado** después de validar un posible cliente: acordar definición de abandono y contrato de datos; obtener de forma legítima un dataset representativo y preparado, analizar su calidad, diseñar entrenamiento con partición por tiempo y entidad, comparar con una línea base y evaluar métricas en un test posterior jamás usado para seleccionar el modelo. Solo después plantear un piloto limitado con controles de seguridad adecuados.

## Despliegue y regresión

Esta versión **no añade migración SQL** ni modifica `render.yaml`, el `Dockerfile.demo`, Keycloak o la lógica del predictor Telco. Se añade un panel informativo en la consola privada v2.7 que explica el contrato y muestra el comando offline. No hay input CSV nuevo en el navegador.

Tras aprobar y fusionar el PR, el procedimiento local será el habitual (no se debe ejecutar hasta entonces):

```powershell
cd C:\Users\arnau\proyectos\saas-churn-mlops
git pull --ff-only
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d --build app
docker compose --env-file .env.staging.local -f compose.private-staging.yml ps
```

Conserva los volúmenes existentes. **No repitas las migraciones 0004/0005** ni ejecutes `docker compose down --volumes`.

Las pruebas de CI verifican el conjunto ficticio, las particiones disjuntas, los requisitos de madurez, la validación estricta de columnas/valores, el rechazo de datos identificativos evidentes, la salida sin identificadores y el comando CLI. Las pruebas privadas de navegador verifican la presencia del contrato y la ausencia de subida de archivos en el panel.
