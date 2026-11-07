"""Fixed synthetic holdout diagnostics: no commercial metrics or saved rows."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from churn_mlops import saas_diagnostics as d
from churn_mlops import saas_experiment as e


def test_v29_monthly_errors_and_capacity_reconcile_with_v28_holdout():
    report = e.run_synthetic_experiment()
    assert report["experiment"] == "synthetic_saas_churn_v29"
    diagnostics = report["held_out_test_diagnostics"]
    assert diagnostics["synthetic_only"] is True
    assert diagnostics["not_real_world_model_validation"] is True
    assert diagnostics["threshold_source"] == "validation_only"
    assert diagnostics["test_rows"] == 300
    assert diagnostics["test_churn_events"] == 44
    errors = diagnostics["classification_errors"]
    assert sum(errors.values()) == 300
    assert errors["true_positive"] + errors["false_negative"] == 44
    assert errors["false_positive"] + errors["true_negative"] == 256
    assert math.isclose(
        errors["true_positive"] / 44,
        report["held_out_test_model"]["recall"],
        abs_tol=1e-6,
    )

    months = diagnostics["monthly_stability"]
    assert [month["snapshot_date"] for month in months] == [
        "2025-10-01", "2025-11-01", "2025-12-01",
    ]
    assert all(month["rows"] == 100 for month in months)
    assert sum(month["observed_churn"] for month in months) == 44
    assert sum(month["errors"]["false_positive"] for month in months) == errors["false_positive"]
    assert sum(month["errors"]["false_negative"] for month in months) == errors["false_negative"]
    assert sum(month["errors"]["true_positive"] for month in months) == errors["true_positive"]
    assert sum(month["errors"]["true_negative"] for month in months) == errors["true_negative"]
    assert all(0 <= month["brier_loss"] <= 1 for month in months)
    assert all(0 <= month["roc_auc"] <= 1 for month in months)

    subgroups = diagnostics["activity_subgroups"]
    assert [group["group"] for group in subgroups] == [
        "activity_0_to_10_days", "activity_11_to_30_days",
    ]
    assert sum(group["rows"] for group in subgroups) == 300
    for key in errors:
        assert sum(group["errors"][key] for group in subgroups) == errors[key]

    capacity = diagnostics["predefined_contact_capacity"]
    assert [entry["capacity_pct"] for entry in capacity] == [5, 10, 20]
    assert [entry["contact_slots"] for entry in capacity] == [15, 30, 60]
    assert all(
        entry["observed_churn_in_selected"] <= entry["contact_slots"]
        and entry["prevented_cancellations_not_measured"]
        for entry in capacity
    )
    top20 = capacity[-1]
    assert top20["observed_churn_in_selected"] == report[
        "held_out_test_model"]["top_20pct_churn_events"]
    assert top20["precision_at_capacity"] == report[
        "held_out_test_model"]["precision_at_top_20pct"]
    assert top20["recall_at_capacity"] == report[
        "held_out_test_model"]["recall_at_top_20pct"]

    text = json.dumps(diagnostics)
    assert "synthetic_0000001" not in text
    assert "account_id" not in text
    assert "prediction_by_account" not in text
    assert "saved_model" not in text
    assert report == e.run_synthetic_experiment()


def test_capacity_ranking_and_confusion_counts_are_precise_and_fixed():
    result = d.diagnose_synthetic_holdout(
        np.array([1, 0, 1, 0]),
        np.array([.9, .8, .2, .1]),
        np.array(["2025-10-01"] * 4),
        np.array([1, 8, 11, 20]),
        validation_selected_threshold=.5,
    )
    assert result["classification_errors"] == {
        "true_positive": 1, "false_positive": 1,
        "false_negative": 1, "true_negative": 1,
    }
    assert result["monthly_stability"][0]["observed_churn"] == 2
    assert result["activity_subgroups"][0]["rows"] == 2
    assert result["activity_subgroups"][1]["rows"] == 2
    first = result["predefined_contact_capacity"][0]
    assert first["contact_slots"] == 1
    assert first["observed_churn_in_selected"] == 1
    assert first["precision_at_capacity"] == 1
    assert first["recall_at_capacity"] == .5
    assert first["random_selection_expected_churn"] == .5
    assert first["observed_lift_vs_random_expectation"] == 2.0


@pytest.mark.parametrize("bad_args", [
    ([1, 0], [.5], ["2025-10-01"] * 2, [0, 3], .5),
    ([1, 0], [.5, float("nan")], ["2025-10-01"] * 2, [0, 3], .5),
    ([1, 2], [.5, .3], ["2025-10-01"] * 2, [0, 3], .5),
    ([1, 0], [.5, .3], ["2025-10-01"] * 2, [0, 31], .5),
    ([1, 0], [.5, .3], ["2025-10-01"] * 2, [0, 3], 0),
    ([1, 0], [.5, .3], ["INVALID"] * 2, [0, 3], .5),
])
def test_diagnostics_reject_invalid_arrays_or_threshold(bad_args):
    y, p, dates, activity, threshold = bad_args
    with pytest.raises(ValueError):
        d.diagnose_synthetic_holdout(
            y, p, dates, activity, validation_selected_threshold=threshold,
        )


def test_one_class_month_and_all_zero_labels_have_no_fabricated_auc():
    result = d.diagnose_synthetic_holdout(
        [0, 0, 0, 0],
        [.1, .2, .2, .3],
        ["2025-10-01"] * 4,
        [20, 20, 20, 20],
        validation_selected_threshold=.5,
    )
    month = result["monthly_stability"][0]
    assert month["roc_auc"] is None
    assert month["pr_auc"] is None
    assert month["observed_churn_pct"] == 0
    assert result["predefined_contact_capacity"][0]["recall_at_capacity"] is None
    assert result["predefined_contact_capacity"][0][
        "observed_lift_vs_random_expectation"
    ] is None


def test_cli_emits_only_aggregate_report_and_writes_no_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/run_saas_experiment.py"), "--demo"]
    completed = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")}, timeout=40,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(completed.stdout)
    assert data["held_out_test_diagnostics"]["test_rows"] == 300
    assert "synthetic_0000001" not in completed.stdout
    assert list(tmp_path.iterdir()) == []
