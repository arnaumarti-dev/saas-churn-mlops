import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn_mlops.config import ARTIFACTS_DIR, BEST_MODEL_PATH

logger = logging.getLogger(__name__)



def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_features),
            (
                "cat",
                Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]),
                categorical_features,
            ),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    baseline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    advanced = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    return {"logistic_regression": baseline, "xgboost": advanced}


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def save_confusion_matrix(cm: list[list[int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def log_and_save(model_name: str, model: Pipeline, metrics: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = ARTIFACTS_DIR / f"{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    cm_path = ARTIFACTS_DIR / f"{model_name}_confusion_matrix.png"
    save_confusion_matrix(metrics["confusion_matrix"], cm_path)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({"model": model_name})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "confusion_matrix"})
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(cm_path))

    logger.info("Saved metrics at %s", metrics_path)


def persist_best_model(model: Pipeline) -> None:
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BEST_MODEL_PATH)
    logger.info("Best model saved to %s", BEST_MODEL_PATH)
