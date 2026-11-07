# Churn Lab v3.2 — panel visual privado del experimento SaaS

**Ámbito: exclusivamente staging privado.** No es un modelo comercial. No hay carga de CSV, predicciones de clientes ni cambios en la demo pública o el predictor IBM Telco.

## Qué muestra

La consola privada incluye un nuevo apartado «Evaluación del modelo SaaS» con los resultados **reales del generador reproducible de 1.200 cuentas ficticias (semilla 28)**. La información se calcula con el evaluador offline de v3.1 en el contenedor local. No se extrae del historial de Acme Demo ni de la empresa seleccionada.

- **Indicadores:** cuentas ficticias, ROC-AUC de prueba, recall y avisos de cambio de distribución.
- **Rendimiento:** conmutador ROC-AUC / PR-AUC; barras que comparan el modelo de regresión logística y la línea base constante sobre el mismo test ficticio. Incluye precisión y tamaños de las tres particiones.
- **Calidad:** tabla de cinco características con medias de entrenamiento y prueba y un PSI orientativo. Cero avisos NO significa datos reales validados.
- **Robustez:** tres escenarios hipotéticos sobre el conjunto de validación únicamente; cambios medios en las puntuaciones y cuentas clasificadas por encima del umbral. No son intervenciones ni predicen cancelaciones evitadas.
- **Accesibilidad:** pestañas navegables por teclado (flechas, Inicio, Fin), descripciones textuales de las barras, diseño adaptable a móvil y preferencia de movimiento reducido.

## API de acceso privado y aislamiento

El nuevo GET /saas/v3/demo-evaluation usa exactamente el mismo control de activación y sesión autenticada que la consola privada. Con el staging desactivado devuelve 404; con el staging activado y sin sesión, 401. No acepta parámetros ni identificadores de empresa. Tampoco recibe datos externos, archivos o entradas del usuario.

El backend genera las 1.200 cuentas ficticias y ejecuta evaluate_offline_saas_pilot con la semilla fija y fecha de observación 2026-03-01. Publica únicamente una lista blanca de métricas agregadas, características agregadas y los tres escenarios de sensibilidad. El informe agregado se guarda en una caché de memoria por proceso para evitar repetir entrenamientos al abrir el panel. **No se guardan modelos ni predicciones por cuenta.**

La aplicación solo muestra datos de demostración, incluso si existe una empresa seleccionada. La interfaz comprueba que el origen esté declarado como ficticio antes de renderizarlo. No almacena resultados en cookies o localStorage. El código JavaScript construye elementos con nodos DOM y textContent, nunca inserta respuestas mediante innerHTML.

## Cómo probarlo en Windows

Después de fusionar el PR, desde la raíz del proyecto:

~~~powershell
cd C:\Users\arnau\proyectos\saas-churn-mlops
git pull --ff-only
docker compose --env-file .env.staging.local -f compose.private-staging.yml up -d --build
docker compose --env-file .env.staging.local -f compose.private-staging.yml ps
~~~

Abre https://app.localhost/saas/v3/console, inicia sesión y verifica CHURN LAB · V3.2. La primera visita calcula la evaluación ficticia; las siguientes reutilizan los agregados en memoria.

**No requiere migraciones SQL ni reiniciar o borrar los volúmenes de PostgreSQL.** No uses down --volumes para actualizar.

## Límites y pruebas

Las métricas corresponden a un mecanismo de cancelación fabricado. ROC-AUC, PR-AUC, PSI y sensibilidad no prueban comportamiento en clientes SaaS reales, causalidad, ahorro, cumplimiento RGPD ni viabilidad comercial.

Los tests verifican acceso autenticado, estado desactivado, cabeceras no-store, estructura de datos agregados, ausencia de identificadores, renderizado de cifras, controles de métricas, tabla de calidad, escenarios de robustez y navegación con teclado en escritorio y móvil. Son controles técnicos, no una auditoría de seguridad comercial.

El trabajo offline de datos SaaS autorizados sigue siendo independiente de la consola web; no se ha creado ninguna vía para introducir CSV reales.
