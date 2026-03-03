import logging

from fastapi import FastAPI, HTTPException

from churn_mlops.inference import load_model, predict
from churn_mlops.logging_config import configure_logging
from churn_mlops.schemas import ChurnInput, ChurnPrediction

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="SaaS Churn Prediction API", version="1.0.0")
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

    probability, label = predict(app.state.model, payload.model_dump())
    return ChurnPrediction(churn_probability=probability, churn_prediction=label)
