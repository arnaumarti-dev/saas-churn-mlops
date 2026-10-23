import logging

import mlflow
import pandas as pd

from churn_mlops.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, RAW_DATA_PATH
from churn_mlops.data import clean_data, engineer_features, load_data, split_train_validation_test
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

    df = engineer_features(clean_data(load_data(RAW_DATA_PATH)))
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(df)

    # Preprocessing and imputation are fitted on training data only.
    candidate_models = build_models(build_preprocessor(X_train))
    best_model_name = ""
    best_auc = float("-inf")

    for model_name, model in candidate_models.items():
        logger.info("Training candidate model: %s", model_name)
        model.fit(X_train, y_train)
        validation_metrics = evaluate_model(model, X_val, y_val)
        log_and_save(model_name, model, validation_metrics, split="validation")

        if validation_metrics["roc_auc"] > best_auc:
            best_auc = validation_metrics["roc_auc"]
            best_model_name = model_name

        logger.info("%s validation metrics: %s", model_name, validation_metrics)

    if not best_model_name:
        raise RuntimeError("No model was trained.")

    # Select by validation only; refit the winning algorithm on train+validation.
    X_dev = pd.concat([X_train, X_val])
    y_dev = pd.concat([y_train, y_val])
    final_model = build_models(build_preprocessor(X_dev))[best_model_name]
    final_model.fit(X_dev, y_dev)

    # Evaluate the held-out test set exactly once after model selection.
    test_metrics = evaluate_model(final_model, X_test, y_test)
    log_and_save(best_model_name, final_model, test_metrics, split="test")
    persist_best_model(final_model)

    logger.info("Selected model: %s (validation ROC-AUC=%.4f)", best_model_name, best_auc)
    logger.info("Final held-out test metrics: %s", test_metrics)


if __name__ == "__main__":
    train()
