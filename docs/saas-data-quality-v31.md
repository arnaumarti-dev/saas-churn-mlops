# Churn Lab v3.1 — calidad y robustez de evaluación SaaS offline

**Estado:** herramienta local, exploratoria y sin despliegue de un modelo SaaS. No es una auditoría RGPD, una certificación de estabilidad ni una demostración de resultados comerciales.

## Qué añade sobre v3.0

El pipeline de entrenamiento con cinco variables numéricas, separación cronológica y umbral fijado en validación se mantiene. El conjunto final de test se utiliza para **una única evaluación de rendimiento**, nunca para seleccionar el modelo o el umbral. El informe JSON incorpora:

**`data_quality`** — Estadísticas **agregadas** por variable y partición (cantidad de filas, mínimo, mediana, máximo y media), porcentajes de actividad nula, MRR cero y 15 o más incidencias en un mes, y comparación de distribución de validación/test respecto a entrenamiento. Esta última usa PSI con intervalos derivados del entrenamiento; `0.2` activa un aviso **heurístico de revisión**, no un diagnóstico estadístico validado. Cuando la característica de referencia es prácticamente constante no calculamos PSI: aparece `null`. Un PSI bajo **no garantiza** estabilidad ni generalización. Los valores ausentes, ilegibles o fuera del contrato se **rechazan**; no hay imputación silenciosa.

**`validation_robustness`** — Tres escenarios **hipotéticos**, calculados solo sobre las 200 cuentas de validación con el modelo y umbral ya fijados: cinco días menos de actividad en los últimos 30 días, dos tickets más de soporte, o una reducción del 20 % del MRR. Los atributos alterados se limitan a los rangos del contrato. Solo se muestran puntuaciones medias, variación media en puntos porcentuales, número agregado por encima del umbral y coincidencia entre los grupos de mayor riesgo. **No se modifican las etiquetas** y la función no usa el test ni reentrena el modelo. Esto mide sensibilidad matemática de puntuaciones, no cambios causales en cancelaciones.

**`--format text`** — Resumen legible en español con número de cuentas, separación temporal, ROC-AUC, PR-AUC, precisión, recall, avisos de calidad y cantidad de pruebas de estrés. **La salida JSON continúa siendo la opción predeterminada** y mantiene sus métricas anteriores.

## Ejecución sin clientes reales

En Windows PowerShell, con Python 3.11 y las dependencias ya instaladas:

```powershell
cd C:\Users\arnau\proyectos\saas-churn-mlops
$env:PYTHONPATH="src"
python scripts/evaluate_saas_pilot.py --demo --as-of 2026-03-01 --format text
```

Para el JSON agregado completo, omite `--format text`. Si prefieres probarlo desde el contenedor privado sin instalar Python, utiliza un comando que invoque el módulo directamente:

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml exec -T app python -c "from datetime import date; from churn_mlops.saas_experiment import generate_experiment_rows; from churn_mlops.saas_pilot_training import evaluate_offline_saas_pilot; from churn_mlops.saas_quality import format_summary; print(format_summary(evaluate_offline_saas_pilot(generate_experiment_rows(), as_of=date(2026,3,1), origin='generated_demo')))"
```

Esto utiliza **1.200 cuentas ficticias**. Los resultados no representan clientes reales, ni estiman la eficacia de contactar cuentas, ni miden dinero ahorrado.

## Reglas de seguridad y metodología

- No hay nuevos endpoints HTTP, cargas de CSV por navegador, cambios en PostgreSQL, Keycloak ni Render.
- El modelo IBM Telco y la consola con historial ficticio siguen operando independientemente.
- El procesamiento local de CSV **sigue requiriendo** `--confirm-authorized-offline-data`, la verificación externa de que existan autorizaciones y un entorno adecuado, y conservar el archivo fuera de GitHub.
- Los reportes pueden contener **estadísticas empresariales confidenciales**, especialmente mínimos y máximos, aunque no tengan identificadores personales; no deben compartirse sin permiso.
- Las distribuciones se inspeccionan **después** de fijar modelo/umbral; la inspección del test no debe utilizarse para cambiar decisiones y volver a reportar las mismas métricas como rendimiento independiente.
- Los escenarios de estrés no sustituyen tests de causalidad, ensayos de intervención o pruebas de generalización.
- Los cinco indicadores numéricos aceptados por el contrato pueden ocultar problemas de procedencia o leakage que ningún test automático puede descartar.

## Pruebas incluidas

Determinismo, estadísticas por partición, PSI alto con perturbaciones artificiales, PSI indefinido para columnas constantes, rechazo de valores inválidos, no mutación del conjunto original, tres escenarios exclusivamente de validación, resumen por CLI y ausencia de escritura de archivos con datos. CI verifica tanto el JSON como el nuevo formato de texto.

**Siguiente avance recomendado sin contactos comerciales:** un flujo local visual para comparar ejecuciones ficticias **sin persistir datos**, con gestión robusta de cambios en el esquema antes de desarrollar un producto comercial.
