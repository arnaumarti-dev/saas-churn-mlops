"""Offline SaaS pilot: never expose rows/IDs or use holdout to train."""
from __future__ import annotations

from datetime import date
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from churn_mlops.pilot_readiness import DataContractError
from churn_mlops.saas_experiment import FEATURES, generate_experiment_rows
from churn_mlops import saas_pilot_training as training

ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 3, 1)


def test_authorized_offline_pipeline_demo_evaluates_without_row_data():
    result = training.evaluate_offline_saas_pilot(
        generate_experiment_rows(), as_of=AS_OF, origin="generated_demo"
    )
    assert result == training.evaluate_offline_saas_pilot(
        generate_experiment_rows(), as_of=AS_OF, origin="generated_demo"
    )
    assert result["workflow"] == "offline_saas_pilot_v31"
    assert result["origin_declared"] == "generated_demo"
    assert result["origin_not_independently_verified"]
    assert result["model_not_saved_or_deployed"]
    assert result["not_production_ready"]
    assert result["not_evidence_of_avoided_churn"]
    assert result["rows"] == 1200
    assert result["threshold_source"] == "validation_only"
    assert (result["splits"]["training"]["rows"],
            result["splits"]["validation"]["rows"],
            result["splits"]["test"]["rows"]) == (700, 200, 300)
    assert result["held_out_test_model"]["rows"] == 300
    assert result["held_out_test_model"]["churn_events"] == 44
    assert result["held_out_test_constant_baseline"]["roc_auc"] == .5
    assert result["held_out_test_model"]["roc_auc"] > .5
    assert result["features"] == list(FEATURES)
    assert "account_id" not in result["features"]
    assert "snapshot_date" not in result["features"]
    assert "churn_next_30d" not in result["features"]
    assert result["splits"]["training"]["last_snapshot"] < (
        result["splits"]["validation"]["first_snapshot"])
    assert result["splits"]["validation"]["last_snapshot"] < (
        result["splits"]["test"]["first_snapshot"])
    serialized = json.dumps(result)
    assert "synthetic_0000001" not in serialized
    assert "customer" not in serialized.lower()
    assert "per_account" not in serialized


def test_model_fit_and_threshold_never_see_holdout_or_identifiers(monkeypatch):
    fits, choices, predicts = [], [], []
    old_fit = training.Pipeline.fit
    old_choose = training._choose_threshold
    old_predict = training.Pipeline.predict_proba

    def inspect_fit(self, X, y, **kwargs):
        fits.append((list(X.columns), len(y)))
        return old_fit(self, X, y, **kwargs)

    def inspect_choose(y, p):
        choices.append((len(y), len(p)))
        return old_choose(y, p)

    def inspect_predict(self, X, **kwargs):
        predicts.append((list(X.columns), len(X)))
        return old_predict(self, X, **kwargs)

    monkeypatch.setattr(training.Pipeline, "fit", inspect_fit)
    monkeypatch.setattr(training.Pipeline, "predict_proba", inspect_predict)
    monkeypatch.setattr(training, "_choose_threshold", inspect_choose)
    training.evaluate_offline_saas_pilot(
        generate_experiment_rows(), as_of=AS_OF, origin="generated_demo"
    )
    assert fits == [(list(FEATURES), 700)]
    assert choices == [(200, 200)]
    assert predicts == ([(list(FEATURES), 200), (list(FEATURES), 300)]
                        + [(list(FEATURES), 200)] * 3)  # Only validation is stressed


def test_protects_temporal_boundaries_and_fails_closed():
    sample = generate_experiment_rows()
    for item in sample:
        if item["snapshot_date"] == "2025-08-01":
            item["snapshot_date"] = "2025-07-10"
    with pytest.raises(DataContractError, match="30 días"):
        training.evaluate_offline_saas_pilot(
            sample, as_of=AS_OF, origin="generated_demo"
        )


def test_blocks_immature_labels_duplicate_ids_bad_columns_and_untrusted_origin():
    data = generate_experiment_rows()
    with pytest.raises(DataContractError, match="insuficientes o inmaduros"):
        training.evaluate_offline_saas_pilot(
            data, as_of=date(2025, 12, 10), origin="generated_demo"
        )
    duplicate = [dict(row) for row in data]
    duplicate[1]["account_id"] = duplicate[0]["account_id"]
    with pytest.raises(DataContractError, match="duplicada"):
        training.evaluate_offline_saas_pilot(
            duplicate, as_of=AS_OF, origin="local_authorized_unverified"
        )
    extra = [dict(row) for row in data]
    extra[0]["email"] = "secret@example.test"
    with pytest.raises(DataContractError, match="columnas"):
        training.evaluate_offline_saas_pilot(
            extra, as_of=AS_OF, origin="local_authorized_unverified"
        )
    with pytest.raises(ValueError, match="Unknown offline data origin"):
        training.evaluate_offline_saas_pilot(data, as_of=AS_OF, origin="network_upload")


def _execute(*args, cwd=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate_saas_pilot.py"), *args],
        cwd=cwd or ROOT, env=env, capture_output=True, text=True, timeout=50,
    )


def test_cli_demo_discloses_unverified_status_and_writes_nothing(tmp_path):
    response = _execute("--demo", "--as-of", "2026-03-01", cwd=tmp_path)
    assert response.returncode == 0, response.stderr
    report = json.loads(response.stdout)
    assert report["origin_declared"] == "generated_demo"
    assert report["not_production_ready"]
    assert "synthetic_0000001" not in response.stdout
    assert list(tmp_path.iterdir()) == []


def test_csv_is_opt_in_and_generates_only_aggregate_metrics(tmp_path):
    csv_file = tmp_path / "authorized_example.csv"
    rows = generate_experiment_rows()
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original_size = csv_file.stat().st_size

    denied = _execute("--csv", str(csv_file), "--as-of", "2026-03-01", cwd=tmp_path)
    assert denied.returncode != 0
    assert "confirm-authorized-offline-data" in denied.stderr

    allowed = _execute(
        "--csv", str(csv_file), "--as-of", "2026-03-01",
        "--confirm-authorized-offline-data", cwd=tmp_path,
    )
    assert allowed.returncode == 0, allowed.stderr
    report = json.loads(allowed.stdout)
    assert report["origin_declared"] == "local_authorized_unverified"
    assert report["origin_not_independently_verified"]
    assert report["not_production_ready"]
    assert "synthetic_0000001" not in allowed.stdout
    assert str(csv_file) not in allowed.stdout
    assert csv_file.stat().st_size == original_size
    assert list(tmp_path.iterdir()) == [csv_file]

    # Fail without printing the arbitrary free-text field or local path.
    with csv_file.open("a", encoding="utf-8") as handle:
        handle.write("\nsecret@example.test\n")
    invalid = _execute(
        "--csv", str(csv_file), "--as-of", "2026-03-01",
        "--confirm-authorized-offline-data", cwd=tmp_path,
    )
    assert invalid.returncode == 2
    assert "secret@example.test" not in invalid.stderr
    assert str(csv_file) not in invalid.stderr
