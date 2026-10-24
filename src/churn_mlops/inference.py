"""Model artifact loading and single/batch churn inference."""
import logging

import joblib
import numpy as np
import pandas as pd

from churn_mlops.config import BEST_MODEL_PATH

logger = logging.getLogger(__name__)


def load_model():
    # joblib uses pickle; only load model artifacts built by our trusted pipeline.
    logger.info("Loading model from %s", BEST_MODEL_PATH)
    return joblib.load(BEST_MODEL_PATH)


def predict_many(artifact, payloads: list[dict]) -> list[tuple[float, int, float]]:
    """Vectorized probabilities using the validation-selected threshold.

    Older artifacts containing only the estimator still default to 0.50.
    """
    if not payloads:
        raise ValueError("A batch must contain at least one customer.")

    if isinstance(artifact, dict):
        model = artifact["model"]
        threshold = float(artifact["decision_threshold"])
    else:
        model = artifact
        threshold = 0.5

    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid saved decision threshold.")

    data = pd.DataFrame(payloads)
    data["AvgMonthlySpend"] = data["TotalCharges"] / data["tenure"].replace(0, 1)
    data["IsNewCustomer"] = (data["tenure"] <= 12).astype(int)
    probas = np.asarray(model.predict_proba(data))
    if probas.shape != (len(payloads), 2):
        raise ValueError("Unexpected probability matrix from the model.")

    results = []
    for probability in probas[:, 1]:
        p = float(probability)
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid probability returned by the model.")
        results.append((p, int(p >= threshold), threshold))
    return results


def predict(artifact, payload: dict) -> tuple[float, int, float]:
    """Single-customer API retained for backward compatibility."""
    return predict_many(artifact, [payload])[0]
