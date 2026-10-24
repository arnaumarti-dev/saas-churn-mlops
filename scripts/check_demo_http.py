"""Exercise the public HTTP contract of a running self-contained demo image.

Run with: DEMO_BASE_URL=http://127.0.0.1:8000 python scripts/check_demo_http.py
No local dataset, API dependencies, or model artifact are required.
"""
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.environ.get("DEMO_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
SAMPLE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.5,
    "TotalCharges": 179,
}


def fetch(request: Request) -> tuple[int, str, bytes]:
    with urlopen(request, timeout=12) as response:
        return response.status, response.headers.get("content-type", ""), response.read()


def main() -> None:
    for attempt in range(30):
        try:
            status, _, _ = fetch(Request(BASE_URL + "/health"))
            if status == 200:
                break
        except (URLError, TimeoutError):
            pass
        time.sleep(2)
    else:
        raise AssertionError(f"Demo API did not become healthy at {BASE_URL}")

    status, content_type, html = fetch(Request(BASE_URL + "/"))
    assert status == 200
    assert "text/html" in content_type, content_type
    assert b"Churn Lab" in html and b'fetch("/predict"' in html

    request = Request(
        BASE_URL + "/predict",
        data=json.dumps(SAMPLE).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    status, content_type, body = fetch(request)
    assert status == 200, body
    assert "application/json" in content_type
    prediction = json.loads(body)
    p = prediction["churn_probability"]
    threshold = prediction["decision_threshold"]
    assert 0 <= p <= 1 and 0 < threshold < 1
    assert prediction["churn_prediction"] == int(p >= threshold)
    print(f"Self-contained demo is operational: {prediction}")


if __name__ == "__main__":
    main()
