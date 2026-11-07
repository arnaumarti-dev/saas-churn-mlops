"""Run the SaaS pilot model entirely OFFLINE — never an upload endpoint.

Fictional self-test:
  PYTHONPATH=src python scripts/evaluate_saas_pilot.py --demo --as-of 2026-03-01

Optional, only with APPROVED local, pseudonymized, authorized CSV:
  PYTHONPATH=src python scripts/evaluate_saas_pilot.py \
    --csv /path/to/approved.csv --as-of 2026-03-01 \
    --confirm-authorized-offline-data

Both modes return only aggregate metrics. The real CSV option does not
establish data legitimacy or an adequate processing environment.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import re
import sys

from churn_mlops.pilot_readiness import DataContractError
from churn_mlops.saas_experiment import generate_experiment_rows
from churn_mlops.saas_pilot_training import evaluate_offline_saas_pilot
from churn_mlops.saas_quality import format_summary
from check_saas_pilot import read_local_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluación SaaS offline: datos ficticios o CSV local autorizado"
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--demo", action="store_true",
                        help="1.200 cuentas ficticias reproducibles")
    inputs.add_argument("--csv", type=str,
                        help="CSV local autorizado, nunca subir a GitHub/Render")
    parser.add_argument("--format", choices=("json", "text"), default="json",
                        help="JSON agregado (por defecto) o resumen legible en consola")
    parser.add_argument("--as-of", required=True,
                        help="Fecha YYYY-MM-DD que cubre las etiquetas posteriores")
    parser.add_argument("--confirm-authorized-offline-data", action="store_true",
                        help="Confirmación explícita antes de leer un CSV local")
    args = parser.parse_args(argv)

    if args.csv and not args.confirm_authorized_offline_data:
        parser.error(
            "Para --csv necesitas --confirm-authorized-offline-data; "
            "no uses datos reales sin autorizaciones y controles adecuados"
        )
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.as_of):
        parser.error("--as-of debe ser YYYY-MM-DD")
    try:
        observed_until = date.fromisoformat(args.as_of)
    except ValueError:
        parser.error("--as-of no es una fecha válida")

    try:
        if args.demo:
            rows = generate_experiment_rows()
            origin = "generated_demo"
        else:
            from pathlib import Path
            rows = read_local_csv(Path(args.csv))
            origin = "local_authorized_unverified"
        result = evaluate_offline_saas_pilot(
            rows, as_of=observed_until, origin=origin
        )
    except DataContractError as exc:
        # Only sanitised errors written by our validator, never input rows/paths.
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except Exception:
        # A library exception may contain original values or paths.
        print(json.dumps({
            "error": "No se ha completado la evaluación local; "
                     "revisa el entorno y el contrato de datos sin publicar el CSV"
        }, ensure_ascii=False), file=sys.stderr)
        return 2

    if args.format == "text":
        print(format_summary(result))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
