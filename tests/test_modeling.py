import pandas as pd

from churn_mlops.modeling import build_preprocessor


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
