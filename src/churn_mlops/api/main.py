import logging
from pathlib import Path
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from churn_mlops.api.security import SecurityMiddleware
from churn_mlops.api.staging_login import router as staging_login_router
from churn_mlops.api.staging_management import router as staging_management_router
from churn_mlops.api.oidc_postgres import (
    Principal,
    get_oidc_principal,
    membership_role,
)
from churn_mlops.api.tenant_auth import (
    ServiceIdentity,
    check_tenant_access,
    get_service_identity,
)

from churn_mlops.inference import load_model, predict, predict_many
from churn_mlops.logging_config import configure_logging
from churn_mlops.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BatchSummary,
    ChurnInput,
    ChurnPrediction,
)

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="SaaS Churn Prediction API", version="1.2.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_HTML = STATIC_DIR / "index.html"
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
app.add_middleware(SecurityMiddleware)
app.include_router(staging_login_router)
app.include_router(staging_management_router)
app.state.model = None


@app.on_event("startup")
def startup_event() -> None:
    try:
        app.state.model = load_model()
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.warning("Model artifact not found. Run training first.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(payload: ChurnInput) -> ChurnPrediction:
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    probability, label, threshold = predict(app.state.model, payload.model_dump())
    return ChurnPrediction(
        churn_probability=probability, churn_prediction=label, decision_threshold=threshold
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    return _score_batch(payload)


def _score_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Evaluate up to 200 demonstrative customer profiles in one model call.

    Inputs and outputs are processed in memory only; no CSV or customer rows
    are saved by this application.
    """
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    customers = [customer.model_dump() for customer in payload.customers]
    values = predict_many(app.state.model, customers)
    predictions = [
        ChurnPrediction(
            churn_probability=probability,
            churn_prediction=label,
            decision_threshold=threshold,
        )
        for probability, label, threshold in values
    ]
    high_risk = sum(item.churn_prediction for item in predictions)
    high_risk_charges = sum(
        customer.MonthlyCharges
        for customer, prediction in zip(payload.customers, predictions)
        if prediction.churn_prediction == 1
    )
    summary = BatchSummary(
        total_customers=len(predictions),
        high_risk_customers=high_risk,
        low_risk_customers=len(predictions) - high_risk,
        average_churn_probability=(
            sum(item.churn_probability for item in predictions) / len(predictions)
        ),
        decision_threshold=predictions[0].decision_threshold,
        high_risk_monthly_charges=round(high_risk_charges, 2),
    )
    return BatchPredictionResponse(predictions=predictions, summary=summary)


@app.get("/saas/v1/tenants/{tenant_id}/me")
def prototype_membership(
    tenant_id: UUID,
    identity: ServiceIdentity = Depends(get_service_identity),
) -> dict[str, str]:
    """Test-only membership inspection: no accounts or tenant data stored."""
    check_tenant_access(tenant_id, identity)
    return {
        "tenant_id": str(identity.tenant_id),
        "actor_id": str(identity.actor_id),
        "role": identity.role,
        "mode": "prototype-only",
    }


@app.post(
    "/saas/v1/tenants/{tenant_id}/predict/batch",
    response_model=BatchPredictionResponse,
)
def prototype_tenant_batch(
    tenant_id: UUID,
    payload: BatchPredictionRequest,
    identity: ServiceIdentity = Depends(get_service_identity),
) -> BatchPredictionResponse:
    """Authorize access before invoking the existing in-memory scoring core."""
    check_tenant_access(tenant_id, identity, can_predict=True)
    return _score_batch(payload)


@app.get("/saas/v2/tenants/{tenant_id}/me")
def oidc_tenant_membership(
    tenant_id: UUID,
    principal: Principal = Depends(get_oidc_principal),
) -> dict[str, str]:
    """Opt-in OIDC access token + PostgreSQL role, staging ONLY."""
    role = membership_role(tenant_id, principal)
    return {"tenant_id": str(tenant_id), "role": role, "mode": "staging-only"}


@app.post(
    "/saas/v2/tenants/{tenant_id}/predict/batch",
    response_model=BatchPredictionResponse,
)
def oidc_tenant_predict_batch(
    tenant_id: UUID,
    payload: BatchPredictionRequest,
    principal: Principal = Depends(get_oidc_principal),
) -> BatchPredictionResponse:
    """Only server-authorized tenant members with scoring roles can predict."""
    role = membership_role(tenant_id, principal)
    if role not in {"owner", "analyst"}:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return _score_batch(payload)


@app.get("/")
def root():
    """Serve a self-contained interactive portfolio demo from this API."""
    return FileResponse(DEMO_HTML, media_type="text/html")
