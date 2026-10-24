from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class ChurnInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0, le=1200)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: FiniteFloat = Field(..., ge=0, le=1_000_000)
    TotalCharges: FiniteFloat = Field(..., ge=0, le=1_000_000_000)


class ChurnPrediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    decision_threshold: float = Field(..., gt=0, lt=1)

# Keep batches intentionally small to limit compute and avoid collecting customer data.
MAX_BATCH_CUSTOMERS = 200


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[ChurnInput] = Field(..., min_length=1, max_length=MAX_BATCH_CUSTOMERS)


class BatchSummary(BaseModel):
    total_customers: int
    high_risk_customers: int
    low_risk_customers: int
    average_churn_probability: float = Field(..., ge=0, le=1)
    decision_threshold: float = Field(..., gt=0, lt=1)
    high_risk_monthly_charges: float = Field(..., ge=0)


class BatchPredictionResponse(BaseModel):
    predictions: list[ChurnPrediction]
    summary: BatchSummary
