"""The SaaS model remains a separate, fictional, offline experiment only."""
from __future__ import annotations

from datetime import date
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from churn_mlops import saas_experiment as e
from churn_mlops.pilot_readiness import COLUMNS, check_pilot_readiness

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_rows_are_reproducible_and_follow_v27_data_contract():
    rows = e.generate_experiment_rows()
    assert rows == e.generate_experiment_rows()
    assert rows != e.generate_experiment_rows(seed=e.SEED + 1)
    assert len(rows) == 1200
    assert all(set(row) == set(COLUMNS) for row in rows)
    assert len({row["account_id"] for row in rows}) == 1200
    contract = check_pilot_readiness(rows, as_of=date(2026, 3, 1))
    assert contract["ready_for_exploratory_pilot"]
    assert contract["snapshot_periods"] == 12
    assert not contract["blockers"]
    assert 0 < contract["positive_labels"] < 1200


def test_chronological_splits_do_not_share_accounts_or_dates():
    rows = e.generate_experiment_rows()
    train, validation, test = e._slice_by_time(pd.DataFrame(rows))
    assert (len(train), len(validation), len(test)) == (700, 200, 300)
    assert train["snapshot_date"].max() < validation["snapshot_date"].min()
    assert validation["snapshot_date"].max() < test["snapshot_date"].min()
    for a, b in ((train, validation), (train, test), (validation, test)):
        assert not set(a["account_id"]) & set(b["account_id"])
        assert not set(a["snapshot_date"]) & set(b["snapshot_date"])


def test_report_is_aggregate_only_compares_with_training_prevalence_baseline():
    result = e.run_synthetic_experiment()
    assert result == e.run_synthetic_experiment()  # fixed random seed
    assert result["synthetic_only"] is True
    assert result["not_validated_for_real_saas"] is True
    assert result["contract"] == "saas-subscriptions-v1"
    assert result["model"] == "logistic_regression_training_only"
    assert set(result["features"]) == set(e.FEATURES)
    assert not (set(result["features"]) & set(e.EXCLUDED_COLUMNS))
    assert 0 < result["selected_threshold_from_validation_only"] < 1
    assert sum(result["splits"][p]["rows"] for p in ("training", "validation", "test")) == 1200
    test = result["held_out_test_model"]
    baseline = result["held_out_test_constant_baseline"]
    validation = result["validation_model"]
    assert test["rows"] == baseline["rows"] == 300
    assert validation["rows"] == 200
    assert test["churn_events"] == baseline["churn_events"]
    assert baseline["roc_auc"] == 0.5
    for section in (test, baseline, validation):
        for metric in ("roc_auc", "pr_auc", "brier_loss",
                       "calibration_ece_5_bins", "precision", "recall", "f1",
                       "precision_at_top_20pct", "recall_at_top_20pct"):
            assert 0 <= section[metric] <= 1
            assert math.isfinite(section[metric])
        assert section["top_20pct_count"] == math.ceil(section["rows"] * .2)
        assert 0 <= section["top_20pct_churn_events"] <= section["churn_events"]
    serialized = json.dumps(result)
    assert "synthetic_0000001" not in serialized
    assert "account_id" not in result["features"]
    assert "snapshot_date" not in result["features"]
    assert "churn_next_30d" not in result["features"]
    assert "customer_predictions" not in result
    assert "trained_model_path" not in result


def test_threshold_is_selected_from_validation_data_not_test(monkeypatch):
    original = e._choose_threshold
    seen = []

    def spy_choose(y, probabilities):
        seen.append((len(y), len(probabilities)))
        return original(y, probabilities)

    monkeypatch.setattr(e, "_choose_threshold", spy_choose)
    result = e.run_synthetic_experiment()
    assert seen == [(200, 200)]
    assert result["held_out_test_model"]["rows"] == 300


def test_model_fit_never_receives_ids_dates_or_future_labels(monkeypatch):
    original_fit = e.Pipeline.fit
    seen = []

    def spy_fit(self, X, y=None, **kwargs):
        seen.append((list(X.columns), len(X), len(y)))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(e.Pipeline, "fit", spy_fit)
    e.run_synthetic_experiment()
    assert seen == [(list(e.FEATURES), 700, 700)]


def test_experiment_fails_closed_with_single_class_or_bad_probabilities():
    with pytest.raises(ValueError, match="both observed label classes"):
        e._scores(np.zeros(30, dtype=int), np.zeros(30), 0.5)
    with pytest.raises(ValueError, match="Invalid probability"):
        e._scores(
            np.array([0, 1, 0, 1]),
            np.array([0.2, 0.8, np.nan, 0.1]),
            0.5,
        )


def test_cli_only_runs_explicit_demo_and_does_not_write_artifacts(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    command = [sys.executable, str(ROOT / "scripts/run_saas_experiment.py")]
    denied = subprocess.run(
        command + ["--csv", "never-send-actual-customers.csv"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert denied.returncode != 0
    ok = subprocess.run(
        command + ["--demo"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert ok.returncode == 0, ok.stderr
    result = json.loads(ok.stdout)
    assert result["synthetic_only"] is True
    assert "synthetic_0000001" not in ok.stdout
    assert list(tmp_path.iterdir()) == []
