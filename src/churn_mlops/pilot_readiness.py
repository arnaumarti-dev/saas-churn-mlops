"""Local-only SaaS pilot dataset readiness checks, with a deterministic demo.

This module DOES NOT train or deploy a SaaS churn model, accept uploads, or
claim that a Telco model is validated for SaaS. Use only with authorized,
properly pseudonymized offline CSVs; never with unapproved personal data.
"""
from __future__ import annotations

from datetime import date, timedelta
from math import isfinite
import re
from typing import Iterable, Mapping

CONTRACT_VERSION = "saas-subscriptions-v1"
HORIZON_DAYS = 30
COLUMNS = (
    "account_id", "snapshot_date", "tenure_months", "mrr_eur",
    "licensed_seats", "active_days_30d", "support_tickets_30d",
    "churn_next_30d",
)
MAX_ROWS = 20_000
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class DataContractError(ValueError):
    """Schema error that never embeds input values or account identifiers."""


def _safe_date(value: object, column: str, row_number: int) -> date:
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise DataContractError(f"Fila {row_number}: formato de {column} inválido")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DataContractError(
            f"Fila {row_number}: fecha de {column} inválida"
        ) from exc


def _whole(value: object, column: str, row: int, lowest: int, highest: int) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"\d+", value):
        raise DataContractError(f"Fila {row}: {column} debe ser un entero")
    number = int(value)
    if not lowest <= number <= highest:
        raise DataContractError(f"Fila {row}: {column} fuera de límites")
    return number


def _money(value: object, row: int) -> float:
    if not isinstance(value, str) or not re.fullmatch(r"\d+(?:\.\d{1,2})?", value):
        raise DataContractError(f"Fila {row}: mrr_eur debe tener hasta 2 decimales")
    number = float(value)
    if not isfinite(number) or not 0 <= number <= 1_000_000:
        raise DataContractError(f"Fila {row}: mrr_eur fuera de límites")
    return number


def check_pilot_readiness(
    rows: Iterable[Mapping[str, str]], *, as_of: date
) -> dict:
    """Validate one snapshot per pseudonymous account and score *data readiness*.

    A clean result does not validate model predictive performance, legal basis,
    source integrity, or whether input features were truly known at snapshot.
    """
    if not isinstance(as_of, date):
        raise ValueError("as_of debe ser una fecha")
    cleaned: list[tuple[date, int]] = []
    identifiers: set[str] = set()
    immature = 0

    for number, raw in enumerate(rows, start=2):
        if number > MAX_ROWS + 1:
            raise DataContractError("Demasiadas filas: límite de 20000")
        if set(raw) != set(COLUMNS):
            raise DataContractError(
                f"Fila {number}: columnas desconocidas o faltantes"
            )
        account = raw["account_id"]
        if not isinstance(account, str) or not _ID_PATTERN.fullmatch(account):
            raise DataContractError(
                f"Fila {number}: account_id debe ser un token opaco sin datos personales"
            )
        if account in identifiers:
            raise DataContractError(
                f"Fila {number}: cuenta duplicada; se permite una fila por cuenta"
            )
        identifiers.add(account)
        snapshot = _safe_date(raw["snapshot_date"], "snapshot_date", number)
        _whole(raw["tenure_months"], "tenure_months", number, 0, 600)
        _money(raw["mrr_eur"], number)
        _whole(raw["licensed_seats"], "licensed_seats", number, 1, 100_000)
        _whole(raw["active_days_30d"], "active_days_30d", number, 0, 30)
        _whole(raw["support_tickets_30d"], "support_tickets_30d", number, 0, 10_000)
        label = _whole(raw["churn_next_30d"], "churn_next_30d", number, 0, 1)
        if snapshot + timedelta(days=HORIZON_DAYS) > as_of:
            immature += 1
        cleaned.append((snapshot, label))

    if not cleaned:
        raise DataContractError("No hay filas para evaluar")

    by_date = sorted({snapshot for snapshot, _ in cleaned})
    # Entire snapshot dates go to one split, never splitting the same date.
    train_end = max(1, int(len(by_date) * 0.60))
    val_end = min(len(by_date), max(train_end + 1, int(len(by_date) * 0.80)))
    groups = (
        ("training", set(by_date[:train_end])),
        ("validation", set(by_date[train_end:val_end])),
        ("test", set(by_date[val_end:])),
    )
    splits = {}
    for name, dates in groups:
        labels = [label for snapshot, label in cleaned if snapshot in dates]
        splits[name] = {
            "rows": len(labels),
            "churn_events": sum(labels),
            "churn_rate_pct": round(100 * sum(labels) / len(labels), 1)
            if labels else None,
            "first_snapshot": min(dates).isoformat() if dates else None,
            "last_snapshot": max(dates).isoformat() if dates else None,
        }

    blockers = []
    if len(cleaned) < 180:
        blockers.append("Se recomiendan al menos 180 cuentas independientes")
    if len(by_date) < 6:
        blockers.append("Se requieren al menos seis fechas de corte distintas")
    if immature:
        blockers.append(
            f"{immature} filas no tienen la ventana futura de 30 días observada"
        )
    for name, values in splits.items():
        if values["rows"] < 30:
            blockers.append(f"{name}: menos de 30 cuentas")
        if values["churn_events"] < 5:
            blockers.append(f"{name}: menos de cinco cancelaciones")

    return {
        "contract_version": CONTRACT_VERSION,
        "ready_for_exploratory_pilot": not blockers,
        "not_model_validation": True,
        "rows": len(cleaned),
        "snapshot_periods": len(by_date),
        "label_horizon_days": HORIZON_DAYS,
        "label_maturity_as_of": as_of.isoformat(),
        "positive_labels": sum(label for _, label in cleaned),
        "splits": splits,
        "blockers": blockers,
        "limitations": [
            "No se entrena ningún modelo ni se mide su precisión predictiva",
            "No comprueba la procedencia, el consentimiento ni la anonimización real",
            "El origen debe demostrar que las variables preceden al corte temporal",
            "El periodo de 30 días y los umbrales son supuestos de trabajo",
        ],
    }


def generate_demo_rows() -> list[dict[str, str]]:
    """Create 240 deterministic, fictional account snapshots across 12 months."""
    examples: list[dict[str, str]] = []
    for month_index in range(12):
        snapshot = date(2025, month_index + 1, 1).isoformat()
        for index in range(20):
            account = month_index * 20 + index + 1
            examples.append({
                "account_id": f"demo_{account:06d}",
                "snapshot_date": snapshot,
                "tenure_months": str(1 + (index * 5 + month_index) % 70),
                "mrr_eur": f"{35 + (index * 7 + month_index * 3) % 140:.2f}",
                "licensed_seats": str(1 + (index * 3 + month_index) % 70),
                "active_days_30d": str(2 + (index * 4 + month_index) % 29),
                "support_tickets_30d": str((index + month_index) % 7),
                "churn_next_30d": str(int((index + month_index * 3) % 5 == 0)),
            })
    return examples
