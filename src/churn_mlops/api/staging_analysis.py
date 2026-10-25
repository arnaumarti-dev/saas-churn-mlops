"""Opt-in per-organization *synthetic* churn scoring and metadata history.

Only bundled, fictitious profiles can be scored in this endpoint. The API
neither accepts nor stores customer rows, identifiers, CSVs or predictions.
Server-owned summary metadata is inserted under PostgreSQL tenant RLS.
"""
from __future__ import annotations

from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from churn_mlops.api.oidc_postgres import Principal
from churn_mlops.api.staging_login import _login_config
from churn_mlops.api.staging_management import _check_origin, _v18_principal
from churn_mlops.schemas import BatchPredictionRequest, BatchSummary

router = APIRouter(prefix="/saas/v3")

# Curated fictional data, never personal/customer information. These specific
# records are fixed by the server, not submitted by an untrusted browser.
_BASE = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5, "TotalCharges": 179.0,
}
SYNTHETIC_PROFILES = BatchPredictionRequest(customers=[
    _BASE,
    {
        **_BASE, "gender": "Male", "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "InternetService": "DSL", "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes", "DeviceProtection": "Yes",
        "TechSupport": "Yes", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
    },
    {
        **_BASE, "SeniorCitizen": 1, "tenure": 12,
        "MultipleLines": "Yes", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "One year",
        "MonthlyCharges": 105.0, "TotalCharges": 1260.0,
    },
])


class EmptySyntheticRun(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _database():
    try:
        import psycopg
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return psycopg


def _tenant_connection(principal: Principal, tenant_id: UUID):
    """Open transaction-local RLS context; caller owns lifecycle of conn."""
    psycopg = _database()
    cfg = _login_config()
    try:
        connection = psycopg.connect(cfg.database_url, connect_timeout=4)
        # psycopg starts an implicit transaction on the next execute.
        connection.execute(
            """SELECT set_config('app.tenant_id', %s, true),
                      set_config('app.oidc_issuer', %s, true),
                      set_config('app.oidc_subject', %s, true)""",
            (str(tenant_id), principal.issuer, principal.subject),
        )
        return connection
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc


def _active_role(connection, principal: Principal, tenant_id: UUID) -> str:
    # RLS ensures this membership SELECT only exposes the caller's own row.
    row = connection.execute(
        """SELECT role FROM saas.memberships
           WHERE tenant_id = %s AND oidc_issuer = %s
             AND oidc_subject = %s AND revoked_at IS NULL""",
        (tenant_id, principal.issuer, principal.subject),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=403, detail="Organization access denied")
    return row[0]


@router.get("/organizations/{tenant_id}/runs")
def list_synthetic_runs(
    tenant_id: UUID, principal: Principal = Depends(_v18_principal),
):
    psycopg = _database()
    try:
        with _tenant_connection(principal, tenant_id) as conn:
            _active_role(conn, principal, tenant_id)
            rows = conn.execute(
                """SELECT id, total_profiles, over_threshold, created_at
                   FROM saas.sandbox_runs
                   WHERE tenant_id = %s ORDER BY created_at DESC, id DESC
                   LIMIT 20""",
                (tenant_id,),
            ).fetchall()
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {
        "tenant_id": str(tenant_id), "synthetic_only": True,
        "runs": [
            {"id": str(run_id), "total_profiles": total,
             "high_risk_customers": high_risk, "created_at": created.isoformat()}
            for run_id, total, high_risk, created in rows
        ],
    }


@router.post("/organizations/{tenant_id}/runs/sample", status_code=201)
def score_synthetic_sample(
    tenant_id: UUID,
    payload: EmptySyntheticRun,
    request: Request,
    principal: Principal = Depends(_v18_principal),
):
    _check_origin(request)
    from churn_mlops.api.main import _score_batch

    psycopg = _database()
    try:
        with _tenant_connection(principal, tenant_id) as conn:
            role = _active_role(conn, principal, tenant_id)
            if role not in ("owner", "analyst"):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            # Score only the fixed examples. A missing model fails closed (503),
            # and must not insert a fake/stale history record.
            result = _score_batch(SYNTHETIC_PROFILES)
            summary: BatchSummary = result.summary
            run_id = uuid4()
            conn.execute(
                """INSERT INTO saas.sandbox_runs
                   (id, tenant_id, created_by_subject, total_profiles, over_threshold)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    run_id, tenant_id, principal.subject,
                    summary.total_customers, summary.high_risk_customers,
                ),
            )
    except psycopg.Error as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    return {
        "id": str(run_id), "tenant_id": str(tenant_id),
        "synthetic_only": True,
        "summary": summary.model_dump(),
    }
