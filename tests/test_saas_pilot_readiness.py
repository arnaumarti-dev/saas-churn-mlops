"""SaaS pilot readiness contract: synthetic-only tests, no private customer data."""
from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from churn_mlops.pilot_readiness import (
    COLUMNS, DataContractError, check_pilot_readiness, generate_demo_rows,
)

AS_OF = date(2026, 3, 1)
ROOT = Path(__file__).resolve().parents[1]


def test_demo_is_reproducible_and_chronological_not_model_accuracy():
    example = generate_demo_rows()
    assert example == generate_demo_rows()
    assert len(example) == 240
    assert len({item["account_id"] for item in example}) == len(example)
    assert all(set(item) == set(COLUMNS) for item in example)
    report = check_pilot_readiness(example, as_of=AS_OF)
    assert report["contract_version"] == "saas-subscriptions-v1"
    assert report["ready_for_exploratory_pilot"] is True
    assert report["not_model_validation"] is True
    assert report["snapshot_periods"] == 12
    assert report["rows"] == 240
    assert report["positive_labels"] == 48
    assert report["blockers"] == []
    split = report["splits"]
    assert [split[k]["rows"] for k in ("training", "validation", "test")] == [
        140, 40, 60]
    assert sum(split[k]["churn_events"] for k in split) == 48
    assert split["training"]["last_snapshot"] < split["validation"]["first_snapshot"]
    assert split["validation"]["last_snapshot"] < split["test"]["first_snapshot"]
    serialized = json.dumps(report)
    assert "demo_000001" not in serialized
    assert "account_id" not in serialized
    assert "model_accuracy" not in report


@pytest.mark.parametrize(
    "field,value",
    [
        ("account_id", "person@example.com"),
        ("snapshot_date", "2025-02-30"),
        ("tenure_months", "-1"),
        ("mrr_eur", "NaN"),
        ("mrr_eur", "1.234"),
        ("licensed_seats", "0"),
        ("active_days_30d", "31"),
        ("support_tickets_30d", "-1"),
        ("churn_next_30d", "2"),
    ],
)
def test_invalid_rows_fail_closed_and_do_not_echo_values(field, value):
    examples = generate_demo_rows()
    examples[0][field] = value
    with pytest.raises(DataContractError) as result:
        check_pilot_readiness(examples, as_of=AS_OF)
    if len(value) > 2:  # Single digits can legitimately occur in row numbers.
        assert value not in str(result.value)
    assert "demo_000001" not in str(result.value)


def test_rejects_unknown_pii_fields_duplicate_accounts_and_missing_values():
    examples = generate_demo_rows()
    examples[0]["email"] = "secret@example.org"
    with pytest.raises(DataContractError, match="columnas"):
        check_pilot_readiness(examples, as_of=AS_OF)
    del examples[0]["email"]
    examples[1]["account_id"] = examples[0]["account_id"]
    with pytest.raises(DataContractError, match="duplicada"):
        check_pilot_readiness(examples, as_of=AS_OF)
    examples[1]["account_id"] = "demo_repaired"
    del examples[1]["mrr_eur"]
    with pytest.raises(DataContractError, match="columnas"):
        check_pilot_readiness(examples, as_of=AS_OF)


def test_insufficient_or_immature_data_not_reported_as_ready():
    few = generate_demo_rows()[:20]
    report = check_pilot_readiness(few, as_of=AS_OF)
    assert report["ready_for_exploratory_pilot"] is False
    assert report["snapshot_periods"] == 1
    assert report["splits"]["validation"]["rows"] == 0
    assert report["splits"]["test"]["rows"] == 0
    assert report["blockers"]
    recent = check_pilot_readiness(
        generate_demo_rows(), as_of=date(2025, 12, 10))
    assert recent["ready_for_exploratory_pilot"] is False
    assert any("ventana futura" in item for item in recent["blockers"])


def test_insufficient_positive_labels_block_a_pilot():
    rows = generate_demo_rows()
    for item in rows:
        item["churn_next_30d"] = "0"
    report = check_pilot_readiness(rows, as_of=AS_OF)
    assert not report["ready_for_exploratory_pilot"]
    assert all(
        any(label in msg for msg in report["blockers"])
        for label in ("training", "validation", "test")
    )


def run_cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_saas_pilot.py"), *args],
        capture_output=True, text=True, cwd=ROOT, env=env, timeout=15,
    )


def test_cli_demo_json_is_ready_without_records():
    response = run_cli("--demo", "--as-of", "2026-03-01")
    assert response.returncode == 0, response.stderr
    output = json.loads(response.stdout)
    assert output["ready_for_exploratory_pilot"] is True
    assert output["rows"] == 240
    assert "demo_000001" not in response.stdout


def test_csv_requires_explicit_authorization_and_rejects_extra_fields(tmp_path):
    csv_path = tmp_path / "fictional.csv"
    rows = generate_demo_rows()
    columns = list(COLUMNS)
    csv_path.write_text(
        ",".join(columns) + "\n" +
        "\n".join(",".join(r[column] for column in columns) for r in rows) +
        "\n",
        encoding="utf-8",
    )
    rejected = run_cli("--csv", str(csv_path), "--as-of", "2026-03-01")
    assert rejected.returncode != 0
    assert "confirm-authorized-offline-data" in rejected.stderr
    allowed = run_cli(
        "--csv", str(csv_path), "--as-of", "2026-03-01",
        "--confirm-authorized-offline-data")
    assert allowed.returncode == 0, allowed.stderr
    assert json.loads(allowed.stdout)["rows"] == 240
    csv_path.write_text("email," + ",".join(columns) + "\n"
                        "secret@example.org," + ",".join(rows[0][c] for c in columns),
                        encoding="utf-8")
    invalid = run_cli(
        "--csv", str(csv_path), "--as-of", "2026-03-01",
        "--confirm-authorized-offline-data")
    assert invalid.returncode == 2
    assert "secret@example.org" not in invalid.stderr
