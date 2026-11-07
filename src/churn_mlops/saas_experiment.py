"""Offline, fictional SaaS-churn modeling experiment — never a deployed model.

No HTTP endpoints, saved predictions, customer records, model pickle or reuse of
the IBM Telco model. Training labels are produced by a deliberately synthetic
data-generating rule: measured accuracy is NOT evidence about actual SaaS users.
"""
from __future__ import annotations

from datetime import date
from math import ceil, exp
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, fbeta_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_mlops.pilot_readiness import (
    COLUMNS, check_pilot_readiness,
)

EXPERIMENT_NAME = "synthetic_saas_churn_v28"
SEED = 28
HORIZON_AS_OF = date(2026, 3, 1)
FEATURES = (
    "tenure_months", "mrr_eur", "licensed_seats",
    "active_days_30d", "support_tickets_30d",
)
EXCLUDED_COLUMNS = ("account_id", "snapshot_date", "churn_next_30d")


def generate_experiment_rows(*, seed: int = SEED) -> list[dict[str, str]]:
    """1,200 independent fictional accounts; the target is sampled, not copied.

    Churn is generated from a synthetic mechanism correlated with earlier
    activity, support usage and tenure. It cannot quantify real-world lift.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    for month in range(12):
        snapshot = date(2025, month + 1, 1).isoformat()
        for index in range(100):
            number = month * 100 + index + 1
            tenure = int(rng.integers(1, 85))
            mrr = int(rng.integers(30, 450))
            seats = int(rng.integers(1, 60))
            active = int(rng.integers(0, 31))
            support = int(rng.poisson(0.9 + (30 - active) / 22))
            # These invented dependencies produce an intentionally learnable
            # signal. They are ASSUMPTIONS, not discovered SaaS behavior.
            log_odds = (
                -2.35 + 0.105 * (15 - active)
                + 0.045 * (12 - min(tenure, 12))
                + 0.23 * (support - 2)
                + 0.065 * month / 11
            )
            probability = 1 / (1 + exp(-log_odds))
            label = int(rng.random() < probability)
            rows.append({
                "account_id": f"synthetic_{number:07d}",
                "snapshot_date": snapshot,
                "tenure_months": str(tenure),
                "mrr_eur": f"{mrr:.2f}",
                "licensed_seats": str(seats),
                "active_days_30d": str(active),
                "support_tickets_30d": str(support),
                "churn_next_30d": str(label),
            })
    return rows


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float:
    """Equal-width expected calibration error on ONE already-scored split."""
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        in_bin = (p >= lower) & ((p < upper) if index < bins - 1 else (p <= upper))
        if np.any(in_bin):
            error += float(np.mean(in_bin)) * abs(
                float(np.mean(p[in_bin])) - float(np.mean(y[in_bin]))
            )
    return round(error, 6)


def _scores(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int]:
    if len(y) < 1 or len(np.unique(y)) != 2:
        raise ValueError("Each evaluation split needs both observed label classes")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid probability array")
    predicted = (p >= threshold).astype(int)
    top_count = max(1, ceil(len(y) * 0.2))
    top_idx = np.argsort(-p, kind="stable")[:top_count]
    top_hits = int(np.sum(y[top_idx]))
    return {
        "rows": len(y),
        "churn_events": int(np.sum(y)),
        "prevalence_pct": round(100 * float(np.mean(y)), 3),
        "roc_auc": round(float(roc_auc_score(y, p)), 6),
        "pr_auc": round(float(average_precision_score(y, p)), 6),
        "brier_loss": round(float(brier_score_loss(y, p)), 6),
        "calibration_ece_5_bins": _ece(y, p),
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 6),
        "recall": round(float(recall_score(y, predicted, zero_division=0)), 6),
        "f1": round(float(f1_score(y, predicted, zero_division=0)), 6),
        "top_20pct_count": top_count,
        "top_20pct_churn_events": top_hits,
        "precision_at_top_20pct": round(top_hits / top_count, 6),
        "recall_at_top_20pct": round(top_hits / int(np.sum(y)), 6),
    }


def _choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Select threshold from validation labels only, never the hold-out test."""
    candidates = [round(i / 100, 2) for i in range(10, 91, 2)]
    return max(
        candidates,
        key=lambda t: (
            fbeta_score(y, (p >= t).astype(int), beta=2, zero_division=0),
            precision_score(y, (p >= t).astype(int), zero_division=0),
            -abs(t - 0.5),
        ),
    )


def _slice_by_time(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(frame["snapshot_date"].unique().tolist())
    if len(dates) < 6:
        raise ValueError("At least six snapshot periods are required")
    train_end = max(1, int(len(dates) * 0.60))
    val_end = min(len(dates), max(train_end + 1, int(len(dates) * 0.80)))
    groups = [
        frame.loc[frame["snapshot_date"].isin(dates[:train_end])].copy(),
        frame.loc[frame["snapshot_date"].isin(dates[train_end:val_end])].copy(),
        frame.loc[frame["snapshot_date"].isin(dates[val_end:])].copy(),
    ]
    if any(group.empty for group in groups):
        raise ValueError("Chronological split produced an empty partition")
    if not (
        groups[0]["snapshot_date"].max() < groups[1]["snapshot_date"].min()
        < groups[2]["snapshot_date"].min()
    ):
        raise ValueError("Chronological partitions overlap")
    return groups[0], groups[1], groups[2]


def run_synthetic_experiment(*, seed: int = SEED) -> dict:
    """Train ONLY an offline synthetic model and report held-out metrics.

    Threshold is chosen using validation ONLY. Test is evaluated once,
    compared to the constant train-churn-prevalence baseline, not tuned.
    No model or input row is written to disk or exposed by an endpoint.
    """
    rows = generate_experiment_rows(seed=seed)
    contract = check_pilot_readiness(rows, as_of=HORIZON_AS_OF)
    if not contract["ready_for_exploratory_pilot"]:
        raise ValueError("Synthetic data failed pilot-readiness checks")
    if set(rows[0]) != set(COLUMNS):
        raise ValueError("Unexpected synthetic feature contract")

    frame = pd.DataFrame(rows)
    train, validation, test = _slice_by_time(frame)
    train_y = train["churn_next_30d"].astype(int).to_numpy()
    val_y = validation["churn_next_30d"].astype(int).to_numpy()
    test_y = test["churn_next_30d"].astype(int).to_numpy()
    if any(len(np.unique(y)) != 2 for y in (train_y, val_y, test_y)):
        raise ValueError("Each temporal split must contain both churn classes")

    def inputs(group: pd.DataFrame) -> pd.DataFrame:
        # Do NOT include account identifiers, dates or future churn labels.
        return group.loc[:, list(FEATURES)].astype(float)

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1500, random_state=seed)),
    ])
    model.fit(inputs(train), train_y)
    val_p = model.predict_proba(inputs(validation))[:, 1]
    selected_threshold = _choose_threshold(val_y, val_p)

    # NEVER select hyperparameters from test data or refit on test.
    test_p = model.predict_proba(inputs(test))[:, 1]
    train_prevalence = float(np.mean(train_y))
    baseline_p = np.full(len(test_y), train_prevalence, dtype=float)

    result = {
        "experiment": EXPERIMENT_NAME,
        "dataset": "generated_1200_fictional_accounts_v1",
        "synthetic_only": True,
        "not_validated_for_real_saas": True,
        "contract": contract["contract_version"],
        "seed": seed,
        "model": "logistic_regression_training_only",
        "features": list(FEATURES),
        "excluded_columns": list(EXCLUDED_COLUMNS),
        "label_horizon_days": contract["label_horizon_days"],
        "selected_threshold_from_validation_only": selected_threshold,
        "splits": {
            name: {
                "rows": len(group),
                "start": group["snapshot_date"].min(),
                "end": group["snapshot_date"].max(),
                "positive_labels": int(group["churn_next_30d"].astype(int).sum()),
            }
            for name, group in zip(
                ("training", "validation", "test"), (train, validation, test)
            )
        },
        "validation_model": _scores(val_y, val_p, selected_threshold),
        "held_out_test_model": _scores(test_y, test_p, selected_threshold),
        "held_out_test_constant_baseline": _scores(
            test_y, baseline_p, selected_threshold
        ),
        "limitations": [
            "El churn se ha generado con una regla artificial y aleatoria",
            "Los resultados solo describen este experimento inventado",
            "No se ha validado el modelo con empresas SaaS ni datos históricos reales",
            "No se ha demostrado calibración externa ni utilidad económica",
            "No se ha guardado ni desplegado un modelo SaaS",
            "Solo se mide el holdout cronológico sin CV, auditoría de fuente o causalidad",
        ],
    }
    return result
