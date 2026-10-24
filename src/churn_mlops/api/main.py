import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from churn_mlops.api.security import SecurityMiddleware

from churn_mlops.inference import load_model, predict
from churn_mlops.logging_config import configure_logging
from churn_mlops.schemas import ChurnInput, ChurnPrediction

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="SaaS Churn Prediction API", version="1.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_HTML = STATIC_DIR / "index.html"
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
app.add_middleware(SecurityMiddleware)
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

@app.get("/")
def root():
    """Serve a self-contained interactive portfolio demo from this API."""
    return FileResponse(DEMO_HTML, media_type="text/html")
