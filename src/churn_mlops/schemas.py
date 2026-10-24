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
