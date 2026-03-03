import logging

import mlflow

from churn_mlops.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RAW_DATA_PATH
from churn_mlops.data import clean_data, engineer_features, load_data, split_data
from churn_mlops.logging_config import configure_logging
from churn_mlops.modeling import (
    build_models,
    build_preprocessor,
    evaluate_model,
    log_and_save,
    persist_best_model,
)

logger = logging.getLogger(__name__)


def train() -> None:
    configure_logging()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    best_model_name = ""
    best_model = None
    best_auc = -1.0

    for model_name, model in models.items():
        logger.info("Training model: %s", model_name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        log_and_save(model_name, model, metrics)

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model_name = model_name
            best_model = model

        logger.info("%s metrics: %s", model_name, metrics)

    if best_model is None:
        raise RuntimeError("No model was trained.")

    persist_best_model(best_model)
    logger.info("Best model: %s (ROC-AUC=%.4f)", best_model_name, best_auc)


if __name__ == "__main__":
    train()
