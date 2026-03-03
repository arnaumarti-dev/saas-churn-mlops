from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "telco_churn.csv"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MLFLOW_TRACKING_URI = (ROOT_DIR / "mlruns").resolve().as_uri()
EXPERIMENT_NAME = "saas-churn-prediction"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Churn"
