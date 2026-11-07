"""Local-only CSV schema check; no network calls, database access or row logs.

Run with the built-in fictional dataset:
  PYTHONPATH=src python scripts/check_saas_pilot.py --demo --as-of 2026-03-01

Only use --csv for data you are independently authorized to process,
with opaque identifiers and NO personal details; NEVER upload customer
records to the public demo or private staging HTTP endpoints.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
import re
import sys

from churn_mlops.pilot_readiness import (
    COLUMNS, DataContractError, check_pilot_readiness, generate_demo_rows,
)

MAX_FILE_BYTES = 5 * 1024 * 1024


def read_local_csv(path: Path) -> list[dict[str, str]]:
    """Read at most 5 MiB; never print input rows or file path."""
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DataContractError("CSV demasiado grande: máximo 5 MiB")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(COLUMNS) or (
            set(reader.fieldnames) != set(COLUMNS)
        ):
            raise DataContractError(
                "Cabeceras CSV inválidas: utiliza exactamente las columnas del contrato"
            )
        return list(reader)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Comprobar preparación de un CSV SaaS (local, sin predicciones)"
    )
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--demo", action="store_true", help="240 cuentas ficticias")
    sources.add_argument("--csv", type=Path, help="Ruta local de CSV autorizado y seudonimizado")
    parser.add_argument(
        "--as-of", required=True,
        help="Fecha YYYY-MM-DD hasta la que se conocen las cancelaciones",
    )
    parser.add_argument(
        "--confirm-authorized-offline-data", action="store_true",
        help="Confirmo que tengo autorización para procesar este CSV localmente",
    )
    args = parser.parse_args(argv)
    if args.csv and not args.confirm_authorized_offline_data:
        parser.error(
            "--csv requiere --confirm-authorized-offline-data; no envíes datos personales"
        )
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.as_of):
        parser.error("--as-of debe tener formato YYYY-MM-DD")
    try:
        as_of = date.fromisoformat(args.as_of)
    except ValueError:
        parser.error("--as-of no es una fecha válida")
    try:
        rows = generate_demo_rows() if args.demo else read_local_csv(args.csv)
        report = check_pilot_readiness(rows, as_of=as_of)
    except (DataContractError, OSError, UnicodeError, csv.Error) as exc:
        # CSV values/identifiers and user paths are NEVER included in output.
        detail = str(exc) if isinstance(exc, DataContractError) else (
            "No se ha podido leer el CSV local"
        )
        print(json.dumps({"error": detail}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_exploratory_pilot"] else 1


if __name__ == "__main__":
    sys.exit(main())
