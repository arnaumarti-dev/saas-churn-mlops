"""Offline SaaS churn pilot *evaluation* for explicitly authorized local data.

Accepts only the restrictive v2.7 eight-field schema, rejects ambiguous
temporal cutoffs, never uploads, persists, logs or returns account-level
predictions, and never modifies the deployed IBM Telco predictor.

A technically successful result is NOT proof of suitability, legal compliance,
calibration across organizations, economic benefit or causal retention.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_mlops.saas_quality import summarize_quality, validation_stress_scenarios
from churn_mlops.pilot_readiness import (
    COLUMNS, DataContractError, HORIZON_DAYS, check_pilot_readiness,
)
from churn_mlops.saas_experiment import (
    EXCLUDED_COLUMNS, FEATURES, _choose_threshold, _scores, _slice_by_time,
)


def _safe_frame(
    rows: list[Mapping[str, str]], as_of: date
) -> tuple[pd.DataFrame, dict]:
    """The schema must be validated before any string is parsed as a feature."""
    readiness = check_pilot_readiness(rows, as_of=as_of)
    if not readiness["ready_for_exploratory_pilot"]:
        # Never include raw CSV values, IDs or file paths in the exception.
        raise DataContractError(
            "Datos insuficientes o inmaduros para la evaluación local; "
            "ejecuta check_saas_pilot.py para ver bloqueos agregados"
        )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    for name in FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    frame["churn_next_30d"] = frame["churn_next_30d"].astype(int)
    return frame, readiness


def _validate_temporal_separation(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Ensure previous labels are mature before the next feature cutoff.

    For each earlier snapshot S, the 'churn in next 30d' label is known no
    later than S+30 days. Later features can only be evaluated when that
    horizon has elapsed. This is a strict *fail closed* gap rule, not
    an assertion that external data genuinely followed it.
    """
    boundary_pairs = [(train, validation), (validation, test)]
    for earlier, later in boundary_pairs:
        last = date.fromisoformat(earlier["snapshot_date"].max())
        first = date.fromisoformat(later["snapshot_date"].min())
        if last + timedelta(days=HORIZON_DAYS) > first:
            raise DataContractError(
                "Las fechas de entrenamiento, validación y prueba no "
                "respetan los 30 días de maduración entre particiones"
            )


def _requires_both_classes(group: pd.DataFrame, phase: str) -> np.ndarray:
    y = group["churn_next_30d"].to_numpy(dtype=int)
    positives, negatives = int(np.sum(y)), len(y) - int(np.sum(y))
    if positives < 5 or negatives < 5:
        raise DataContractError(
            f"{phase}: se requieren al menos cinco bajas y cinco no-bajas"
        )
    return y


def evaluate_offline_saas_pilot(
    rows: Iterable[Mapping[str, str]],
    *,
    as_of: date,
    origin: str,
) -> dict:
    """Fit and evaluate an exploratory model entirely in process.

    'origin' is supplied by the CLI and is a DECLARATION, not verified data
    provenance. No model or data is written to files or returned to caller.
    """
    if origin not in {"generated_demo", "local_authorized_unverified"}:
        raise ValueError("Unknown offline data origin")
    local_rows = list(rows)
    frame, readiness = _safe_frame(local_rows, as_of)
    train, validation, test = _slice_by_time(frame)
    _validate_temporal_separation(train, validation, test)

    y_train = _requires_both_classes(train, "training")
    y_validation = _requires_both_classes(validation, "validation")
    y_test = _requires_both_classes(test, "test")

    def inputs(group: pd.DataFrame) -> pd.DataFrame:
        # Whitelist-only: no account IDs, snapshot dates, or *future* churn.
        return group.loc[:, list(FEATURES)].astype(float)

    pipeline = Pipeline(steps=[
        ("normalizer", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1500, random_state=42)),
    ])
    pipeline.fit(inputs(train), y_train)

    validation_probabilities = pipeline.predict_proba(inputs(validation))[:, 1]
    threshold = _choose_threshold(y_validation, validation_probabilities)

    # The holdout TEST is never used to choose the algorithm or threshold.
    test_probabilities = pipeline.predict_proba(inputs(test))[:, 1]
    baseline_probability = float(np.mean(y_train))
    baseline = np.full(len(y_test), baseline_probability, dtype=float)

    # Descriptive quality checks never select features, parameters or thresholds.
    # Stress scenarios query VALIDATION only and cannot re-fit the model.
    quality = summarize_quality(train, validation, test)
    robustness = validation_stress_scenarios(
        pipeline, inputs(validation), validation_probabilities,
        decision_threshold=threshold,
    )

    # All model outputs are only aggregate metrics.
    return {
        "workflow": "offline_saas_pilot_v31",
        "origin_declared": origin,
        "origin_not_independently_verified": True,
        "model_not_saved_or_deployed": True,
        "not_production_ready": True,
        "not_evidence_of_avoided_churn": True,
        "contract": readiness["contract_version"],
        "label_horizon_days": HORIZON_DAYS,
        "data_maturity_as_of": as_of.isoformat(),
        "features": list(FEATURES),
        "excluded_from_model": list(EXCLUDED_COLUMNS),
        "rows": len(frame),
        "threshold_source": "validation_only",
        "decision_threshold": threshold,
        "splits": {
            key: {
                "rows": len(group),
                "events": int(group["churn_next_30d"].sum()),
                "first_snapshot": str(group["snapshot_date"].min()),
                "last_snapshot": str(group["snapshot_date"].max()),
            }
            for key, group in zip(
                ("training", "validation", "test"),
                (train, validation, test),
            )
        },
        "validation_model": _scores(
            y_validation, validation_probabilities, threshold
        ),
        "held_out_test_model": _scores(
            y_test, test_probabilities, threshold
        ),
        "held_out_test_constant_baseline": _scores(
            y_test, baseline, threshold
        ),
        "data_quality": quality,
        "validation_robustness": robustness,
        "limitations": [
            "La procedencia y autorización de los datos no se pueden "
            "comprobar automáticamente",
            "Un token opaco puede seguir siendo dato personal seudonimizado",
            "Las variables deben existir realmente antes de cada snapshot",
            "Una división cronológica y un test no prueban generalización",
            "El test se debe reservar para una sola evaluación final, "
            "sin iterar modelos en función de su resultado",
            "No se ha demostrado que contactar con cuentas de riesgo "
            "evite cancelaciones ni genere ingresos",
            "No se ofrece una API SaaS de producción ni un modelo desplegado",
            "PSI y escenarios de estrés son diagnósticos de ingeniería, no certificación estadística",
        ],
    }
