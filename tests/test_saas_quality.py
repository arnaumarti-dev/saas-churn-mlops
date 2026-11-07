"""Offline v3.1 QA: deterministic aggregates and validation-only stress."""
from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from churn_mlops.saas_experiment import FEATURES, _slice_by_time, generate_experiment_rows
from churn_mlops.saas_pilot_training import evaluate_offline_saas_pilot
from churn_mlops.saas_quality import (
    _population_stability, format_summary, summarize_quality,
    validation_stress_scenarios,
)

ROOT = Path(__file__).resolve().parents[1]


def split_demo():
    return _slice_by_time(pd.DataFrame(generate_experiment_rows()))


def test_quality_is_aggregate_per_split_and_deterministic():
    train, validation, test = split_demo()
    q = summarize_quality(train, validation, test)
    assert q == summarize_quality(train, validation, test)
    assert q["schema_missing_values_rejected_before_training"] is True
    assert q["all_input_features_finite"] is True
    assert q["no_statistical_significance_claimed"] is True
    assert list(q["features"]) == list(FEATURES)
    for feature in FEATURES:
        v = q["features"][feature]
        assert v["statistics"]["training"]["rows"] == 700
        assert v["statistics"]["validation"]["rows"] == 200
        assert v["statistics"]["test"]["rows"] == 300
        for stat in v["statistics"].values():
            assert stat["min"] <= stat["median"] <= stat["max"]
            assert stat["min"] <= stat["mean"] <= stat["max"]
    assert sum(q["descriptive_anomalies"][s]["rows"]
               for s in ("training", "validation", "test")) == 1200
    serialized = json.dumps(q)
    assert "synthetic_0000001" not in serialized
    assert "account_id" not in serialized
    assert "churn_next_30d" not in serialized


def test_distribution_shift_detects_large_feature_change_but_not_significance():
    train, val, test = split_demo()
    modified = test.copy()
    modified["active_days_30d"] = 0
    q = summarize_quality(train, val, modified)
    drift = q["features"]["active_days_30d"]["psi_test_vs_training"]
    assert drift is not None and drift >= 0.2
    assert any("active_days_30d" in flag and "prueba" in flag
               for flag in q["distribution_flags"])
    assert q["no_statistical_significance_claimed"] is True
    # No meaningful train quantiles -> leave PSI undefined.
    assert _population_stability(np.ones(100), np.zeros(100)) is None
    with pytest.raises(ValueError, match="Nonfinite"):
        bad = modified.copy()
        bad.loc[bad.index[0], "mrr_eur"] = np.nan
        summarize_quality(train, val, bad)


class DummyModel:
    def __init__(self):
        self.calls = []

    def predict_proba(self, frame):
        self.calls.append((list(frame.columns), len(frame)))
        activity = frame["active_days_30d"].to_numpy(dtype=float)
        probability = np.clip(.05 + activity / 40, .01, .99)
        return np.column_stack((1 - probability, probability))


def test_stress_scenarios_do_not_modify_inputs_or_touch_test_data():
    _, validation, _ = split_demo()
    model = DummyModel()
    inputs = validation.loc[:, list(FEATURES)].astype(float)
    original = inputs.copy(deep=True)
    base_probs = model.predict_proba(inputs)[:, 1]
    report = validation_stress_scenarios(
        model, inputs, base_probs, decision_threshold=.5
    )
    assert inputs.equals(original)
    assert report["evaluated_split"] == "validation_only"
    assert report["model_not_refitted"] and report["test_not_queried_for_stress"]
    assert len(report["scenarios"]) == 3
    assert [item["scenario"] for item in report["scenarios"]] == [
        "activity_5_days_less", "support_2_more_tickets",
        "mrr_20_percent_lower",
    ]
    for scenario in report["scenarios"]:
        assert scenario["validation_rows"] == 200
        assert scenario["hypothetical_only"] is True
        assert 0 <= scenario["same_top_20pct_count"] <= scenario["top_20pct_count"] == 40
    assert model.calls == [(list(FEATURES), 200)] * 4
    assert report["scenarios"][0]["mean_score_change_pp"] < 0
    assert report["scenarios"][1]["mean_score_change_pp"] == 0
    assert report["scenarios"][2]["mean_score_change_pp"] == 0


def test_invalid_score_dimensions_or_threshold_fail_closed():
    _, validation, _ = split_demo()
    inputs = validation.loc[:, list(FEATURES)].astype(float)
    model = DummyModel()
    base = model.predict_proba(inputs)[:, 1]
    with pytest.raises(ValueError, match="dimensions"):
        validation_stress_scenarios(model, inputs, base[:-1], decision_threshold=.5)
    with pytest.raises(ValueError, match="baseline scores"):
        invalid = base.copy()
        invalid[0] = np.nan
        validation_stress_scenarios(model, inputs, invalid, decision_threshold=.5)
    with pytest.raises(ValueError, match="Threshold"):
        validation_stress_scenarios(model, inputs, base, decision_threshold=0)


def test_full_v31_report_is_serializable_and_has_sensitivity_diagnostics():
    report = evaluate_offline_saas_pilot(
        generate_experiment_rows(), as_of=date(2026, 3, 1),
        origin="generated_demo",
    )
    assert report["workflow"] == "offline_saas_pilot_v31"
    assert report["rows"] == 1200
    assert report["validation_robustness"]["evaluated_split"] == "validation_only"
    assert len(report["validation_robustness"]["scenarios"]) == 3
    assert report["data_quality"]["only_aggregate_statistics"] is True
    assert "synthetic_0000001" not in json.dumps(report)
    summary = format_summary(report)
    assert "CHURN LAB V3.1" in summary
    assert "1.200" not in summary or "1200" in summary
    assert "No validado en producción" in summary
    assert "synthetic_" not in summary
    assert "account_id" not in summary
    assert "3 escenarios hipotéticos" in summary


def test_cli_text_mode_readable_and_does_not_write_files(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/evaluate_saas_pilot.py"),
         "--demo", "--as-of", "2026-03-01", "--format", "text"],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        cwd=tmp_path, capture_output=True, text=True, timeout=40,
    )
    assert completed.returncode == 0, completed.stderr
    assert "CHURN LAB V3.1" in completed.stdout
    assert "Control de calidad" in completed.stdout
    assert "Robustez" in completed.stdout
    assert "synthetic_" not in completed.stdout
    assert list(tmp_path.iterdir()) == []


def test_cli_defaults_to_json_and_rejects_unknown_format(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/evaluate_saas_pilot.py"),
               "--demo", "--as-of", "2026-03-01"]
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(command, env=environment, cwd=tmp_path,
                               capture_output=True, text=True, timeout=40)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["workflow"] == "offline_saas_pilot_v31"
    assert report["not_production_ready"]
    rejected = subprocess.run(command + ["--format", "html"], env=environment,
                              cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert rejected.returncode != 0
    assert list(tmp_path.iterdir()) == []
