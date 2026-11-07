"""Offline-only, aggregate SaaS pilot data diagnostics and validation stress tests.

Analyses describe data distributions and hypothetical changes in model scores.
They cannot prove causal effects, production reliability or data legitimacy.
No row-level predictions, identifiers, paths, files or network calls.
"""
from __future__ import annotations

from math import ceil, isfinite
import numpy as np
import pandas as pd

from churn_mlops.saas_experiment import FEATURES

SPLITS = ("training", "validation", "test")
PSI_REVIEW_THRESHOLD = 0.2  # Heuristic for inspection, not a calibrated test.
SCENARIOS = (
    ("activity_5_days_less", "active_days_30d", -5, 0.0, 30.0),
    ("support_2_more_tickets", "support_tickets_30d", 2, 0.0, 10000.0),
    ("mrr_20_percent_lower", "mrr_eur", 0.8, 0.0, 1000000.0),
)


def _values(group: pd.DataFrame, feature: str) -> np.ndarray:
    values = group.loc[:, feature].to_numpy(dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite, empty or missing feature statistics")
    return values


def _population_stability(reference: np.ndarray, observed: np.ndarray) -> float | None:
    """Numeric PSI using equal-frequency reference quantiles; post-hoc only.

    If a reference feature is almost constant, no defensible reference
    quantile bins exist; return null instead of an invented score.
    """
    cuts = np.unique(np.quantile(reference, [0.2, 0.4, 0.6, 0.8]))
    if len(cuts) < 2:
        return None
    bins = np.concatenate(([-np.inf], cuts, [np.inf]))
    ref_counts = np.histogram(reference, bins=bins)[0].astype(float)
    new_counts = np.histogram(observed, bins=bins)[0].astype(float)
    ref_rate = np.maximum(ref_counts / len(reference), 1e-6)
    new_rate = np.maximum(new_counts / len(observed), 1e-6)
    return round(float(np.sum((new_rate - ref_rate) * np.log(new_rate / ref_rate))), 6)


def summarize_quality(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame,
) -> dict:
    """Only aggregate descriptive statistics; test is read for audit, not tuning."""
    groups = {"training": train, "validation": validation, "test": test}
    summaries: dict[str, dict] = {}
    warnings: list[str] = []
    for feature in FEATURES:
        ref = _values(train, feature)
        stats = {}
        for name in SPLITS:
            values = _values(groups[name], feature)
            stats[name] = {
                "rows": int(len(values)),
                "min": round(float(np.min(values)), 3),
                "median": round(float(np.median(values)), 3),
                "max": round(float(np.max(values)), 3),
                "mean": round(float(np.mean(values)), 3),
            }
        val_shift = _population_stability(ref, _values(validation, feature))
        test_shift = _population_stability(ref, _values(test, feature))
        if val_shift is not None and val_shift >= PSI_REVIEW_THRESHOLD:
            warnings.append(f"{feature}: distribución de validación distinta del entrenamiento")
        if test_shift is not None and test_shift >= PSI_REVIEW_THRESHOLD:
            warnings.append(f"{feature}: distribución de prueba distinta del entrenamiento")
        summaries[feature] = {
            "statistics": stats,
            "psi_validation_vs_training": val_shift,
            "psi_test_vs_training": test_shift,
            "review_threshold_heuristic": PSI_REVIEW_THRESHOLD,
        }

    # v2.7 rejects missing/invalid values before this function runs.
    anomalies = {}
    for name, group in groups.items():
        activity = _values(group, "active_days_30d")
        revenue = _values(group, "mrr_eur")
        tickets = _values(group, "support_tickets_30d")
        anomalies[name] = {
            "rows": len(group),
            "zero_activity_pct": round(100 * float(np.mean(activity == 0)), 2),
            "zero_mrr_pct": round(100 * float(np.mean(revenue == 0)), 2),
            "support_15_plus_pct": round(100 * float(np.mean(tickets >= 15)), 2),
        }

    return {
        "schema_missing_values_rejected_before_training": True,
        "all_input_features_finite": True,
        "only_aggregate_statistics": True,
        "features": summaries,
        "descriptive_anomalies": anomalies,
        "distribution_flags": warnings,
        "no_statistical_significance_claimed": True,
        "notes": [
            "PSI >= 0.2 es una regla orientativa, no un umbral validado para estos datos",
            "Un valor PSI bajo no demuestra que exista estabilidad en producción",
            "Las columnas ausentes, celdas vacías y valores no numéricos son rechazados por el contrato",
            "La calidad y procedencia de las variables no se pueden verificar con estadísticas",
        ],
    }


def validation_stress_scenarios(
    pipeline, validation_inputs: pd.DataFrame, base_probabilities: np.ndarray,
    *,
    decision_threshold: float,
) -> dict:
    """Change validation-only inputs AFTER model and threshold are fixed.

    These are sensitivity checks under invented feature changes. They NEVER
    change labels, fit the model, read the test, or quantify churn prevented.
    """
    if list(validation_inputs.columns) != list(FEATURES):
        raise ValueError("Unexpected validation feature list")
    baseline = np.asarray(base_probabilities, dtype=float)
    if baseline.ndim != 1 or len(baseline) != len(validation_inputs) or not len(baseline):
        raise ValueError("Invalid baseline score dimensions")
    if not np.all(np.isfinite(baseline)) or np.any((baseline < 0) | (baseline > 1)):
        raise ValueError("Invalid baseline scores")
    if not isfinite(decision_threshold) or not 0 < decision_threshold < 1:
        raise ValueError("Threshold must be selected on validation first")
    count = len(baseline)
    count_top = max(1, ceil(count * 0.2))
    original_top = set(np.argsort(-baseline, kind="stable")[:count_top].tolist())
    scenarios = []
    for name, feature, parameter, lower, upper in SCENARIOS:
        modified = validation_inputs.copy(deep=True)
        base = modified[feature].to_numpy(dtype=float)
        if name == "mrr_20_percent_lower":
            values = base * parameter
        else:
            values = base + parameter
        modified[feature] = np.clip(values, lower, upper)
        new_p = np.asarray(pipeline.predict_proba(modified)[:, 1], dtype=float)
        if new_p.shape != baseline.shape or not np.all(np.isfinite(new_p)):
            raise ValueError("Invalid scenario scores")
        if np.any((new_p < 0) | (new_p > 1)):
            raise ValueError("Scenario returned out-of-range probabilities")
        stressed_top = set(np.argsort(-new_p, kind="stable")[:count_top].tolist())
        scenarios.append({
            "scenario": name,
            "feature_changed": feature,
            "validation_rows": count,
            "mean_score_pct": round(100 * float(np.mean(new_p)), 3),
            "mean_score_change_pp": round(100 * float(np.mean(new_p - baseline)), 3),
            "risk_flags_above_validation_threshold": int(
                np.sum(new_p >= decision_threshold)
            ),
            "same_top_20pct_count": len(original_top & stressed_top),
            "top_20pct_count": count_top,
            "hypothetical_only": True,
        })
    return {
        "evaluated_split": "validation_only",
        "labels_not_modified": True,
        "model_not_refitted": True,
        "test_not_queried_for_stress": True,
        "base_mean_score_pct": round(100 * float(np.mean(baseline)), 3),
        "scenarios": scenarios,
        "notes": [
            "Las variaciones de atributos son hipotéticas, no tratamientos sobre clientes",
            "La puntuación alterada no implica un cambio causal en el riesgo real",
            "Los umbrales, el modelo y las variables se mantienen congelados",
        ],
    }


def format_summary(report: dict) -> str:
    """Spanish, copy-friendly console summary from aggregate whitelist fields."""
    if report.get("workflow") != "offline_saas_pilot_v31":
        raise ValueError("Unsupported offline report version")
    quality = report["data_quality"]
    robustness = report["validation_robustness"]
    train = report["splits"]["training"]
    val = report["splits"]["validation"]
    test = report["splits"]["test"]
    m = report["held_out_test_model"]
    b = report["held_out_test_constant_baseline"]
    lines = [
        "CHURN LAB V3.1 — EVALUACIÓN OFFLINE",
        "Origen declarado: " + (
            "datos generados de prueba"
            if report["origin_declared"] == "generated_demo"
            else "CSV local autorizado (NO verificado)"
        ),
        "Estado: EXPERIMENTAL · No desplegado · No validado en producción",
        f"Datos: {report['rows']} cuentas | horizonte {report['label_horizon_days']} días",
        f"Entrenamiento: {train['rows']} | Validación: {val['rows']} | Prueba: {test['rows']}",
        "Modelo de prueba: ROC-AUC {:.3f} | PR-AUC {:.3f} | Recall {:.1%} | Precisión {:.1%}".format(
            m["roc_auc"], m["pr_auc"], m["recall"], m["precision"]
        ),
        "Línea base: ROC-AUC {:.3f} | PR-AUC {:.3f}".format(
            b["roc_auc"], b["pr_auc"]
        ),
        f"Control de calidad: {len(quality['distribution_flags'])} avisos de distribución",
        "Robustez: {} escenarios hipotéticos en validación".format(
            len(robustness["scenarios"])
        ),
        "Estos resultados NO prueban precisión comercial ni cancelaciones evitadas.",
    ]
    for warning in quality["distribution_flags"]:
        lines.append("Revisar: " + warning)
    return "\n".join(lines)
