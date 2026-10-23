import pandas as pd

from churn_mlops.data import clean_data, engineer_features, split_train_validation_test


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
    # Imputation must occur inside the model pipeline, after splitting.
    assert cleaned["TotalCharges"].isna().sum() == 1
    assert cleaned.loc[1, "TotalCharges"] == 20.5
    assert set(cleaned["Churn"].unique()) == {0, 1}


def test_engineer_features_adds_expected_columns() -> None:
    df = pd.DataFrame({"tenure": [0, 24], "TotalCharges": [0.0, 240.0]})
    engineered = engineer_features(df)
    assert "AvgMonthlySpend" in engineered.columns
    assert "IsNewCustomer" in engineered.columns
    assert engineered.loc[0, "AvgMonthlySpend"] == 0
    assert engineered.loc[1, "IsNewCustomer"] == 0


def test_train_validation_test_split_is_stratified_and_disjoint() -> None:
    df = pd.DataFrame(
        {
            "tenure": list(range(100)),
            "TotalCharges": [float(i) for i in range(100)],
            "Churn": [0, 1] * 50,
        }
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_validation_test(df)
    assert (len(X_train), len(X_val), len(X_test)) == (60, 20, 20)
    assert (len(y_train), len(y_val), len(y_test)) == (60, 20, 20)
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)
    assert all(y.mean() == 0.5 for y in (y_train, y_val, y_test))
