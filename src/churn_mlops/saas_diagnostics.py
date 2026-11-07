"""Post-hoc diagnostics for a FROZEN synthetic SaaS temporal holdout.

No training, threshold fitting, calibration fitting, financial projections,
individual scores, customer identifiers, file storage or network access.
Every number describes invented examples ONLY.
"""
from __future__ import annotations

from math import ceil

import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

CAPACITY_PERCENTS = (5, 10, 20)


def _finite_arrays(labels, probabilities, snapshot_dates, activity_days, threshold):
    y = np.asarray(labels)
    p = np.asarray(probabilities, dtype=float)
    dates = np.asarray(snapshot_dates, dtype=str)
    activity = np.asarray(activity_days, dtype=float)
    if any(array.ndim != 1 for array in (y, p, dates, activity)):
        raise ValueError("Holdout diagnostics require one-dimensional arrays")
    count = len(y)
    if not count or any(len(array) != count for array in (p, dates, activity)):
        raise ValueError("Holdout diagnostic array lengths must match")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("Invalid binary label")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probability")
    if not np.all(np.isfinite(activity)) or np.any((activity < 0) | (activity > 30)):
        raise ValueError("Invalid activity days")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Decision threshold must have been set on validation")
    if any(len(d) != 10 or d[4] != "-" or d[7] != "-" for d in dates):
        raise ValueError("Invalid snapshot date labels")
    return y.astype(int), p, dates, activity


def _errors(labels, predicted) -> dict[str, int]:
    return {
        "true_positive": int(np.sum((labels == 1) & (predicted == 1))),
        "false_positive": int(np.sum((labels == 0) & (predicted == 1))),
        "false_negative": int(np.sum((labels == 1) & (predicted == 0))),
        "true_negative": int(np.sum((labels == 0) & (predicted == 0))),
    }


def _monthly_metrics(labels, probabilities, dates, threshold) -> list[dict]:
    result = []
    predictions = (probabilities >= threshold).astype(int)
    for month in sorted(set(dates.tolist())):
        mask = dates == month
        y = labels[mask]
        p = probabilities[mask]
        one_class = len(set(y.tolist())) == 1
        errors = _errors(y, predictions[mask])
        result.append({
            "snapshot_date": month,
            "rows": int(len(y)),
            "observed_churn": int(np.sum(y)),
            "observed_churn_pct": round(100 * float(np.mean(y)), 3),
            "mean_prediction_pct": round(100 * float(np.mean(p)), 3),
            "brier_loss": round(float(brier_score_loss(y, p)), 6),
            "roc_auc": None if one_class else round(float(roc_auc_score(y, p)), 6),
            "pr_auc": None if one_class else round(float(average_precision_score(y, p)), 6),
            "errors": errors,
        })
    return result


def _activity_slices(labels, probabilities, activity, threshold) -> list[dict]:
    predictions = (probabilities >= threshold).astype(int)
    groups = (
        ("activity_0_to_10_days", activity <= 10),
        ("activity_11_to_30_days", activity > 10),
    )
    result = []
    for name, mask in groups:
        y = labels[mask]
        result.append({
            "group": name,
            "rows": int(len(y)),
            "observed_churn": int(np.sum(y)),
            "errors": _errors(y, predictions[mask]),
        })
    return result


def _capacity_diagnostics(labels, probabilities) -> list[dict]:
    """Pre-specified contact budgets on the SAME frozen test probabilities.

    Random contact expectation is not a randomized control group, and hits
    do not mean preventable churn. Ranking ties keep stable original order.
    """
    order = np.argsort(-probabilities, kind="stable")
    count = len(labels)
    total_events = int(np.sum(labels))
    reports = []
    for percent in CAPACITY_PERCENTS:
        contacts = ceil(count * percent / 100)
        hits = int(np.sum(labels[order[:contacts]]))
        random_expected = contacts * total_events / count
        reports.append({
            "capacity_pct": percent,
            "contact_slots": contacts,
            "observed_churn_in_selected": hits,
            "precision_at_capacity": round(hits / contacts, 6),
            "recall_at_capacity": round(hits / total_events, 6) if total_events else None,
            "random_selection_expected_churn": round(random_expected, 3),
            "observed_lift_vs_random_expectation": (
                round(hits / random_expected, 3) if random_expected else None
            ),
            "prevented_cancellations_not_measured": True,
        })
    return reports


def diagnose_synthetic_holdout(
    labels,
    probabilities,
    snapshot_dates,
    activity_days,
    *,
    validation_selected_threshold: float,
) -> dict:
    """Evaluate errors/months/capacity without changing any model decision.

    May be called only on the held-out synthetic test set after all model
    choices are frozen. Do not treat the returned metrics as real-world lift.
    """
    y, p, dates, activity = _finite_arrays(
        labels, probabilities, snapshot_dates, activity_days,
        validation_selected_threshold,
    )
    predictions = (p >= validation_selected_threshold).astype(int)
    months = _monthly_metrics(y, p, dates, validation_selected_threshold)
    return {
        "synthetic_only": True,
        "not_real_world_model_validation": True,
        "threshold_source": "validation_only",
        "test_rows": int(len(y)),
        "test_churn_events": int(np.sum(y)),
        "classification_errors": _errors(y, predictions),
        "monthly_stability": months,
        "activity_subgroups": _activity_slices(
            y, p, activity, validation_selected_threshold
        ),
        "predefined_contact_capacity": _capacity_diagnostics(y, p),
        "limitations": [
            "No se ha validado el modelo con suscripciones SaaS reales",
            "Cada mes tiene pocas cuentas ficticias; las diferencias no prueban drift",
            "Las bandas de actividad son descriptivas, no una auditoría de sesgos",
            "Seleccionar casos de riesgo no demuestra retención ni ahorro de ingresos",
            "Los supuestos de capacidad (5, 10, 20 %) son fijos y exploratorios",
            "Ningún umbral o hiperparámetro se selecciona con el conjunto de prueba",
        ],
    }
