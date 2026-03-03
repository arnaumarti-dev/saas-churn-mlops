import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_mlops.config import RANDOM_STATE, TARGET_COL, TEST_SIZE

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop(columns=["customerID"], errors="ignore")
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["TotalCharges"].median())
    cleaned[TARGET_COL] = cleaned[TARGET_COL].map({"Yes": 1, "No": 0})
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["AvgMonthlySpend"] = features["TotalCharges"] / (features["tenure"].replace(0, 1))
    features["IsNewCustomer"] = (features["tenure"] <= 12).astype(int)
    return features


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
