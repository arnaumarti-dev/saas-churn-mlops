import pandas as pd

from churn_mlops.data import clean_data, engineer_features


def test_clean_data_handles_total_charges_and_target() -> None:
    df = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "tenure": [1, 2],
            "TotalCharges": [" ", "20.5"],
            "Churn": ["Yes", "No"],
        }
    )

    cleaned = clean_data(df)
    assert "customerID" not in cleaned.columns
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert set(cleaned["Churn"].unique()) == {0, 1}


def test_engineer_features_adds_expected_columns() -> None:
    df = pd.DataFrame({"tenure": [0, 24], "TotalCharges": [0.0, 240.0]})
    engineered = engineer_features(df)
    assert "AvgMonthlySpend" in engineered.columns
    assert "IsNewCustomer" in engineered.columns
    assert engineered.loc[0, "AvgMonthlySpend"] == 0
    assert engineered.loc[1, "IsNewCustomer"] == 0
