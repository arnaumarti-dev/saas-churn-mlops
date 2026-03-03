import logging

import joblib
import pandas as pd
import numpy as np 

from churn_mlops.config import BEST_MODEL_PATH

logger = logging.getLogger(__name__)



def load_model():
    logger.info("Loading model from %s", BEST_MODEL_PATH)
    return joblib.load(BEST_MODEL_PATH)


def predict(model, payload: dict) -> tuple[float, int]:
    data = pd.DataFrame([payload])
    data["AvgMonthlySpend"] = data["TotalCharges"] / data["tenure"].replace(0, 1)
    data["IsNewCustomer"] = (data["tenure"] <= 12).astype(int)
    probas = np.asarray(model.predict_proba(data))
    probability = float(probas[0, 1])
    label = int(model.predict(data)[0])
    return probability, label
