import logging

import joblib
import pandas as pd
import numpy as np 

from churn_mlops.config import BEST_MODEL_PATH

logger = logging.getLogger(__name__)



def load_model():
    logger.info("Loading model from %s", BEST_MODEL_PATH)
    return joblib.load(BEST_MODEL_PATH)


def predict(artifact, payload: dict) -> tuple[float, int, float]:
    """Predict using the threshold packaged with the trained model.

    Older plain-estimator artifacts continue to work with threshold 0.50.
    """
    if isinstance(artifact, dict):
        model = artifact["model"]
        threshold = float(artifact["decision_threshold"])
    else:
        model = artifact
        threshold = 0.5

    if not 0 < threshold < 1:
        raise ValueError("Invalid saved decision threshold.")

    data = pd.DataFrame([payload])
    data["AvgMonthlySpend"] = data["TotalCharges"] / data["tenure"].replace(0, 1)
    data["IsNewCustomer"] = (data["tenure"] <= 12).astype(int)
    probas = np.asarray(model.predict_proba(data))
    probability = float(probas[0, 1])
    label = int(probability >= threshold)
    return probability, label, threshold
