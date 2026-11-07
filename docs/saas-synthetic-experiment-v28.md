# Churn Lab v2.8 — primer experimento de churn SaaS (100 % sintético)

**Esta versión NO es un modelo comercial.** Sirve exclusivamente para comprobar, de principio a fin, un procedimiento reproducible de entrenamiento y evaluación con ejemplos inventados. No se usa para tomar decisiones sobre personas o empresas.

## Qué hemos construido

- **Generador** determinista de **1.200 cuentas ficticias** y 12 fechas de corte mensuales en 2025. Cumple el contrato `saas-subscriptions-v1` de v2.7. Las cancelaciones se generan a partir de una fórmula inventada que depende, entre otras cosas, de actividad, antigüedad y soporte, más un componente aleatorio.
- **Modelo**: regresión logística con imputación y normalización ajustadas solo en entrenamiento. Entradas permitidas: `tenure_months`, `mrr_eur`, `licensed_seats`, `active_days_30d`, `support_tickets_30d`.
- **Excluidos del modelo**: `account_id`, `snapshot_date` y la etiqueta futura `churn_next_30d`. No deben entrar en el vector predictivo.
- **Partición cronológica** de las 12 fechas: 7 meses para entrenamiento (700 cuentas), 2 para validación (200) y los 3 últimos para test (300). Las cuentas y fechas no se repiten entre los conjuntos.
- **Selección de umbral**: solo sobre validación, maximizando F2, sin ver el test. El test solo se utiliza para la evaluación final.
- **Línea base**: probabilidad constante igual a la tasa de cancelación observada en el entrenamiento. Permite comparar ROC-AUC, PR-AUC y Brier; no es otro modelo sofisticado.
- **Métricas**: ROC-AUC, PR-AUC (average precision), pérdida de Brier, error de calibración ECE (5 intervalos), precisión, recall, F1, número de aciertos entre el 20 % de perfiles con mayor puntuación y cobertura de eventos en ese 20 %.
- **Privacidad**: ejecución *offline*, sin endpoint HTTP, sin lectura de CSV, sin guardar modelo/joblib, sin predicciones por cuenta y sin emitir cuentas individuales en el informe JSON.

### Ejecutar desde Windows PowerShell

Usando tu instalación de Python y las dependencias de `requirements.txt`:

```powershell
cd C:\Users\arnau\proyectos\saas-churn-mlops
$env:PYTHONPATH="src"
python scripts/run_saas_experiment.py --demo
```

El programa imprime en consola un JSON agregado con `validation_model`, `held_out_test_model` y `held_out_test_constant_baseline`. **No requiere Docker ni contacto con el servicio privado.** Debes obtener los valores reales del propio programa; no están fijados en esta documentación.

### Cómo interpretar el resultado

Una cifra alta de ROC-AUC o precision en este experimento solo muestra que el modelo ha aprendido parte del **mecanismo artificial con que nosotros mismos fabricamos las etiquetas**. No acredita predicción de bajas reales, utilidad económica, reducción de churn ni calibración en nuevos negocios.

El "baseline" es la prevalencia de churn en entrenamiento (una probabilidad constante); su ROC-AUC suele ser 0,5 porque no discrimina entre cuentas. El modelo debe evaluarse en un nuevo conjunto histórico representativo antes de sostener que supera una línea base comercial.

Aunque las fechas están ordenadas, **la procedencia de las variables es artificial**. En un piloto real hace falta demostrar que solo se usaron características conocidas al instante de cada corte y etiquetas confirmadas 30 días después. Un indicador agregado limpio no sustituye auditorías, acuerdos con clientes ni revisión del RGPD.

### Qué se mantiene sin cambios

El predictor Telco sigue usando su pipeline y `models/best_model.joblib`. Esta rama no lo sobreescribe ni despliega el nuevo experimento. Las rutas `/predict`, `/predict/batch`, la consola privada, los modelos de v2.6 y la gestión de usuarios no se transforman en inferencia SaaS.

**No añade migraciones PostgreSQL, dependencias nuevas, servicios, formularios de carga de archivos, cuentas, facturación ni cambios de Render.** La demostración pública continúa siendo educativa y se mantiene independiente.

### Siguiente paso hacia monetización

1. Entrevistar a empresas de suscripción reales para acordar una definición operacional de cancelación, horizonte y decisiones de retención.
2. Conseguir datos históricos **autorizados y representativos** mediante un proceso con controles de seguridad, base jurídica adecuada y acuerdos de tratamiento si aplica.
3. Evaluar el pipeline con partición por entidad y tiempo; documentar calidad de datos, calibración, precision@capacidad, costes de intervenir, generalización y sesgos.
4. Solo entonces plantear un piloto de pago con un alcance y criterios de éxito verificables.

**Importante:** v2.8 es un paso de ingeniería experimental. El lanzamiento comercial todavía depende de validación empírica y seguridad.
