# Churn Lab v2.9 — diagnósticos de test sintético: meses, errores y capacidad

**La v2.9 es un experimento offline basado únicamente en datos inventados.** Las métricas no validan un predictor para clientes reales, ni demuestran que una intervención evite bajas. No incluye cargas de CSV reales, facturación, APIs ni despliegues nuevos.

## Qué añade sobre v2.8

Conserva exactamente el entrenamiento experimental de v2.8: 1.200 cuentas inventadas, 700 para entrenamiento, 200 para validación y **300 cuentas de prueba final** en octubre, noviembre y diciembre de 2025. La regresión logística, el preprocesado y el umbral de decisión se eligen **sin utilizar el test**.

La sección nueva del JSON, `held_out_test_diagnostics`, expone solo **agregados** del **mismo test congelado**:

1. **`classification_errors`**: verdaderos positivos, falsos positivos, falsos negativos y verdaderos negativos al umbral elegido en validación. La suma de los cuatro debe coincidir con 300.
2. **`monthly_stability`**: 100 cuentas ficticias por mes, etiquetas positivas, proporción observada, probabilidad media, ROC-AUC, PR-AUC, pérdida de Brier y cuatro clases de errores por mes. Si un mes solo tiene una clase, ROC-AUC/PR-AUC aparecen como `null` en lugar de inventar una cifra.
3. **`activity_subgroups`**: diagnóstico descriptivo en dos bandas de actividad previas al corte (0–10 días y 11–30 días). Solo publica recuentos de errores y etiquetas agregadas; no afirma equivalencia de grupos, ausencia de sesgo ni causalidad.
4. **`predefined_contact_capacity`**: tres presupuestos **fijados antes de mirar el test**: 5 % (15 cuentas), 10 % (30 cuentas) y 20 % (60 cuentas). Mide cuántas **cancelaciones etiquetadas** aparecen entre los casos con mayor puntuación, precisión en ese presupuesto y cobertura de eventos. Compara el resultado con la **expectativa matemática de selección aleatoria** calculada a partir de la prevalencia del test. Esta expectativa no procede de un ensayo aleatorizado.

**Límite clave:** una cuenta marcada como posible cancelación no significa que podamos evitar su baja. No hay medición de tratamientos, clientes retenidos, ingresos recuperados, retorno de inversión ni rendimiento real en producción. El pequeño test de solo tres meses tampoco permite asegurar estabilidad, detectar drift real ni extrapolar a otra industria.

## Cómo ejecutar en Windows

Después de fusionar el PR y actualizar el código, en PowerShell desde el directorio del proyecto:

```powershell
$env:PYTHONPATH="src"
python scripts/run_saas_experiment.py --demo
```

O, si tienes disponible el contenedor de staging con las dependencias del predictor Telco y quieres ejecutarlo **sin instalar Python**:

```powershell
docker compose --env-file .env.staging.local -f compose.private-staging.yml exec -T app python -c "import json; from churn_mlops.saas_experiment import run_synthetic_experiment; print(json.dumps(run_synthetic_experiment(), indent=2, ensure_ascii=False))"
```

El resultado contiene el experimento de v2.8 más `held_out_test_diagnostics`. El nuevo análisis no guarda un modelo ni cuentas, no crea historiales y no modifica PostgreSQL. El resultado puede cambiar si se modifica explícitamente la semilla del generador; por defecto usa la semilla fija 28.

## Pruebas y controles

- Se comprueba que los 12 recuentos de confusión de los tres meses suman exactamente los 4 recuentos de la prueba completa.
- La unión de ambos grupos de actividad equivale a las 300 cuentas de prueba y sus errores agregados.
- La métrica a capacidad del 20 % coincide con la métrica de selección del top 20 % ya existente en v2.8.
- Se rechazan entradas con longitud incongruente, valores no finitos, etiquetas no binarias, fechas mal formadas, actividad fuera de límites o umbrales inválidos.
- No se muestran identificadores, predicciones por cuenta, probabilidades individuales ni ficheros `.joblib`.
- La QA del navegador comprueba que la nueva sección informa expresamente de la naturaleza experimental, pero no crea un formulario para introducir clientes reales.

## Despliegue

No hay migración SQL, nuevas dependencias, cambios en el modelo Telco, los datos de Acme Demo, Keycloak, Dockerfile o Render. La interfaz privada añade una explicación de los diagnósticos, pero **no calcula ni almacena resultados de manera automática por empresa**. El cálculo solo se ejecuta localmente con el comando anterior.

## Siguiente paso realmente comercial

Para vender un piloto necesitamos entrevistar empresas de suscripción que tengan historial autorizado, definir churn y costes de intervención, validar retrospectivamente el modelo con variables conocidas antes de las cancelaciones, y fijar criterios de éxito medibles. Una simulación con datos inventados no sustituye esas pruebas.
