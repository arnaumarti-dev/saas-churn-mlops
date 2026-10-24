import numpy as np
import pandas as pd
import pytest

from churn_mlops.modeling import build_preprocessor, evaluate_model, select_decision_threshold


def test_numeric_imputer_learns_training_median_only() -> None:
    train = pd.DataFrame(
        {
            "TotalCharges": [10.0, 20.0, float("nan")],
            "Contract": ["Monthly", "Yearly", "Monthly"],
        }
    )
    held_out = pd.DataFrame(
        {
            "TotalCharges": [10000.0],
            "Contract": ["Monthly"],
        }
    )
    preprocessor = build_preprocessor(train)
    preprocessor.fit(train)

    numeric_pipeline = preprocessor.named_transformers_["num"]
    assert numeric_pipeline.named_steps["imputer"].statistics_[0] == 15.0
    assert preprocessor.transform(held_out).shape[0] == 1


class StubProbabilityModel:
    """Deterministic probabilities to test threshold selection independently."""

    def predict_proba(self, X):
        positives = X["score"].to_numpy()
        return np.column_stack([1.0 - positives, positives])


def test_tuned_threshold_prefers_recall_with_precision_floor() -> None:
    scores = [0.95, 0.80, 0.46, 0.36, 0.39, 0.27, 0.19, 0.08]
    X_val = pd.DataFrame({"score": scores})
    y_val = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
    model = StubProbabilityModel()

    threshold, metrics = select_decision_threshold(model, X_val, y_val, min_precision=0.55)
    baseline = evaluate_model(model, X_val, y_val, threshold=0.5)

    assert 0.15 <= threshold < 0.5
    assert metrics["precision"] >= 0.55
    assert metrics["recall"] == 1.0
    assert metrics["f2"] > baseline["f2"]
    assert metrics["decision_threshold"] == threshold


def test_no_eligible_threshold_falls_back_to_half() -> None:
    model = StubProbabilityModel()
    X_val = pd.DataFrame({"score": [0.10, 0.90, 0.80]})
    y_val = pd.Series([1, 0, 0])

    threshold, metrics = select_decision_threshold(model, X_val, y_val, min_precision=0.9)
    assert threshold == 0.5
    assert metrics["decision_threshold"] == 0.5


def test_invalid_decision_threshold_is_rejected() -> None:
    model = StubProbabilityModel()
    X_val = pd.DataFrame({"score": [0.1, 0.9]})
    y_val = pd.Series([0, 1])

    with pytest.raises(ValueError):
        evaluate_model(model, X_val, y_val, threshold=1.0)
    with pytest.raises(ValueError):
        select_decision_threshold(model, X_val, y_val, min_precision=0.0)
