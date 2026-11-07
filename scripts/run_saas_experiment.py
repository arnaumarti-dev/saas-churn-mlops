"""Run one completely OFFLINE demonstration with invented SaaS subscriptions.

Windows PowerShell, from the repository root:
  $env:PYTHONPATH="src"
  python scripts/run_saas_experiment.py --demo

Requires only dependencies already pinned in requirements.txt. It does not read
user CSVs, persist models or customer rows, or expose any HTTP service.
"""
from __future__ import annotations

import argparse
import json

from churn_mlops.saas_experiment import run_synthetic_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline synthetic SaaS-churn modeling experiment, not a commercial model"
    )
    parser.add_argument(
        "--demo", action="store_true", required=True,
        help="Use fixed, synthetic example accounts only (no CSV input)",
    )
    args = parser.parse_args(argv)
    assert args.demo
    print(json.dumps(
        run_synthetic_experiment(), ensure_ascii=False, indent=2, allow_nan=False
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
