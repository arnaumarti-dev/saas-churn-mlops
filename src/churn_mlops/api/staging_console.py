"""Default-off, authenticated browser console for private synthetic SaaS staging.

Unlike public /assets this router serves the HTML, CSS and JS only after
the same OIDC session + feature flags required by management operations.
It never embeds issuer/subject, secrets, organization names or invite tokens
into HTML; the browser retrieves current membership through the existing DB.
"""
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from churn_mlops.api.oidc_postgres import Principal
from churn_mlops.api.staging_login import _my_orgs
from churn_mlops.api.staging_management import _v18_principal

router = APIRouter(prefix="/saas/v3")
PRIVATE_ASSETS = Path(__file__).resolve().parent / "private_assets"


@router.get("/console", include_in_schema=False)
def console(_: Principal = Depends(_v18_principal)):
    return FileResponse(
        PRIVATE_ASSETS / "console.html", media_type="text/html",
    )


@router.get("/console.js", include_in_schema=False)
def console_script(_: Principal = Depends(_v18_principal)):
    return FileResponse(
        PRIVATE_ASSETS / "console.js", media_type="application/javascript",
    )


@router.get("/console.css", include_in_schema=False)
def console_styles(_: Principal = Depends(_v18_principal)):
    return FileResponse(
        PRIVATE_ASSETS / "console.css", media_type="text/css",
    )


@lru_cache(maxsize=1)
def _fixed_fictional_evaluation() -> dict:
    """Cache ONLY aggregates, never a model, account rows or customer data.

    One deterministic, locally generated sample (seed 28). Do not add CSV
    sources, request parameters, company IDs or database access here.
    """
    from datetime import date

    from churn_mlops.saas_experiment import FEATURES, generate_experiment_rows
    from churn_mlops.saas_pilot_training import evaluate_offline_saas_pilot

    report = evaluate_offline_saas_pilot(
        generate_experiment_rows(), as_of=date(2026, 3, 1),
        origin="generated_demo",
    )
    quality = report["data_quality"]
    robustness = report["validation_robustness"]
    return {
        "demo_only": True,
        "source": "fixed_seed_28_fictional_accounts",
        "not_real_customer_data": True,
        "not_validated_for_production": True,
        "not_evidence_of_avoided_churn": True,
        "accounts": report["rows"],
        "splits": {
            phase: report["splits"][phase]["rows"]
            for phase in ("training", "validation", "test")
        },
        "model": {
            key: report["held_out_test_model"][key]
            for key in ("roc_auc", "pr_auc", "precision", "recall")
        },
        "baseline": {
            key: report["held_out_test_constant_baseline"][key]
            for key in ("roc_auc", "pr_auc")
        },
        "quality": {
            "distribution_flag_count": len(quality["distribution_flags"]),
            "features": [
                {
                    "name": feature,
                    "training_mean": quality["features"][feature][
                        "statistics"]["training"]["mean"],
                    "test_mean": quality["features"][feature][
                        "statistics"]["test"]["mean"],
                    "psi_test": quality["features"][feature][
                        "psi_test_vs_training"],
                }
                for feature in FEATURES
            ],
        },
        "robustness": {
            "evaluated_split": "validation_only",
            "scenarios": [
                {
                    "scenario": item["scenario"],
                    "mean_score_change_pp": item["mean_score_change_pp"],
                    "risk_flags_above_validation_threshold": item[
                        "risk_flags_above_validation_threshold"],
                    "same_top_20pct_count": item["same_top_20pct_count"],
                    "top_20pct_count": item["top_20pct_count"],
                }
                for item in robustness["scenarios"]
            ],
        },
    }


@router.get("/demo-evaluation", include_in_schema=False)
def demo_evaluation(_: Principal = Depends(_v18_principal)):
    """Authenticated staging-only aggregate report; no user supplied input."""
    return _fixed_fictional_evaluation()


@router.get("/organizations/list")
def list_organizations(principal: Principal = Depends(_v18_principal)):
    # Postgres enforces issuer/subject membership and tenant-specific names;
    # never fetch all organizations or accept a claimed caller-supplied tenant.
    return {"organizations": _my_orgs(principal)}
