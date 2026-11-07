# Churn Lab v3.0 — evaluación SaaS completamente offline

**Objetivo:** permitir pruebas reproducibles con un conjunto ficticio y, cuando exista autorización y un entorno seguro, evaluar **localmente** un CSV histórico con el contrato v2.7. **No es todavía un SaaS comercial ni un modelo apto para producción.** No se contacta a empresas ni se suben datos a la consola.

## Novedad respecto a v2.9

El experimento anterior entrenaba y evaluaba únicamente las 1.200 cuentas ficticias que generaba el propio código. La v3.0 introduce un **evaluador offline separado**, que acepta tanto ese mismo conjunto sintético como un archivo CSV local previamente autorizado. Ejecuta los mismos pasos sin guardar modelos, CSV transformados ni predicciones individuales y sin tocar el predictor IBM Telco desplegado.

- Valida exactamente ocho columnas: `account_id`, `snapshot_date`, `tenure_months`, `mrr_eur`, `licensed_seats`, `active_days_30d`, `support_tickets_30d` y `churn_next_30d`.
- Rechaza campos adicionales como emails y nombres, fechas no válidas, IDs duplicados, etiquetas fuera de rango, más de 20.000 filas o ficheros de más de 5 MiB. **Estas comprobaciones NO garantizan anonimización**, legitimidad ni ausencia de fugas de datos.
- Comprueba el historial de 30 días posterior al corte mediante la fecha `--as-of` y los criterios mínimos de preparación de v2.7.
- Reserva fechas enteras para entrenamiento (aprox. 60 %), validación (20 %) y prueba (20 %); **exige además 30 días de separación entre el último corte de entrenamiento y el primer corte de validación, y entre validación y prueba**. Si faltan esos intervalos, rechaza el conjunto en lugar de utilizar etiquetas que aún no se habrían observado.
- Exige al menos cinco cancelaciones y cinco no-cancelaciones en cada fase.
- Entrena regresión logística con normalización basada **solo en entrenamiento** y utiliza exclusivamente las cinco variables numéricas anteriores al corte. Los IDs, las fechas y la etiqueta futura están excluidos del vector del modelo.
- Selecciona el umbral exclusivamente sobre validación y evalúa el test sin reentrenar. Reporta ROC-AUC, PR-AUC, precisión, recall, F1, Brier, calibración y rendimiento top 20 % frente a la línea base constante de prevalencia en entrenamiento.
- El resultado en consola es **solo JSON agregado**; no imprime identificadores, rutas de archivo, valores de filas, probabilidades por cuenta ni escribe artefactos `.joblib`.

## Probar ahora con datos ficticios

Después de fusionar y actualizar el repositorio, abre PowerShell en la raíz. El comando necesita Python 3.11 y dependencias del proyecto instaladas en un entorno virtual local:

```powershell
cd C:\Users\arnau\proyectos\saas-churn-mlops
$env:PYTHONPATH="src"
python scripts/evaluate_saas_pilot.py --demo --as-of 2026-03-01
```

Si todavía no tienes Python con las dependencias, puedes probar **solo el conjunto ficticio** dentro del contenedor privado existente, sin trasladar ningún CSV:

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml exec -T app python -c "import json; from datetime import date; from churn_mlops.saas_experiment import generate_experiment_rows; from churn_mlops.saas_pilot_training import evaluate_offline_saas_pilot; print(json.dumps(evaluate_offline_saas_pilot(generate_experiment_rows(), as_of=date(2026,3,1), origin='generated_demo'), indent=2, ensure_ascii=False))"
```

El informe incluirá `workflow: offline_saas_pilot_v30`, métricas de validación, test, línea base y advertencias explícitas. La ejecución ficticia debería ser reproducible, pero **sus métricas no se extrapolan al mundo real**.

## Cómo sería una evaluación local autorizada en el futuro

**No necesitas datos reales para completar o probar v3.0.** Si más adelante cuentas con datos históricos, antes de ejecutar esta opción se requieren: autorización del propietario, evaluación del RGPD y de otras normas aplicables, un entorno aislado/controlado, identificadores seudonimizados, una definición de cancelación verificable, variables realmente conocidas antes de cada corte y documentación de la procedencia de las etiquetas.

Con esos requisitos satisfechos, únicamente desde un ordenador autorizado, fuera de GitHub y Render:

```powershell
$env:PYTHONPATH="src"
python scripts/evaluate_saas_pilot.py --csv C:\ruta-segura\suscripciones-autorizadas.csv --as-of 2026-03-01 --confirm-authorized-offline-data
```

Sustituye `--as-of` por la fecha real hasta la que se ha observado el resultado de todas las suscripciones. Esta bandera confirma que **tú declaras** estar autorizado; el programa no puede comprobarlo. El CSV debe permanecer fuera del repositorio y no debe copiarse al contenedor de la demo, a GitHub, a ChatGPT ni a Render.

**El informe puede contener información confidencial agregada de una empresa**, aunque no publique filas; consérvalo localmente y no lo compartas sin permiso.

## Interpretación correcta

Un informe técnicamente válido NO demuestra que el modelo esté listo para venderse. Una única muestra histórica podría contener sesgos, errores de etiqueta, fuga de información no observable para el programa o distribuciones distintas de producción. Además, el test debe mantenerse **cerrado para decisiones de selección posteriores**; si iteramos según sus resultados, necesitamos reservar otro conjunto verdaderamente externo.

La regresión logística ofrece una referencia inicial y el umbral se selecciona con F2 como supuesto técnico, no como una función de costes comerciales. No se calcula causalidad ni bajas evitadas, y todavía faltan seguridad de explotación, observabilidad, retraining con criterios definidos y evaluación en clientes reales.

## Compatibilidad y seguridad

- **No hay cambios SQL**, no se modifica PostgreSQL, OIDC, Keycloak ni el historial de Acme Demo.
- **No hay endpoints nuevos** ni formularios CSV en el navegador. Se añade exclusivamente un apartado informativo sobre el evaluador offline.
- No se modifica `render.yaml`, `Dockerfile.demo` ni el modelo Telco existente.
- El CLI devuelve código 0 si la evaluación finaliza y 2 en errores de validación o entorno, mostrando mensajes que no contienen datos de filas o rutas.
- CI prueba el modo ficticio y los tests verifican rechazo de datos inmaduros, separación temporal mínima, ausencia de fuga de IDs/etiqueta futura, umbral basado únicamente en validación y opt-in explícito del CSV.

**Próximos desarrollos posibles sin contactar a nadie:** mejorar el empaquetado offline, construir una herramienta local de exploración de calidad sin datos personales y añadir pruebas de robustez/datos faltantes, manteniendo el modelo fuera de producción.
