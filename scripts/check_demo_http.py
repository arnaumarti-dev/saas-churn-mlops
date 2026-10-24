"""Exercise the public HTTP contract of a running self-contained demo image.

Run with: DEMO_BASE_URL=http://127.0.0.1:8000 python scripts/check_demo_http.py
No local dataset, API dependencies, or model artifact are required.
"""
import json
import os
import time
from urllib.error import HTTPError
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


def wait_for_health(base_url: str = BASE_URL, attempts: int = 30, delay: float = 2) -> None:
    """Wait for startup, including transient socket resets during ASGI initialization."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    last_error = "no response"
    for attempt in range(attempts):
        try:
            status, _, _ = fetch(Request(base_url + "/health"))
            if status == 200:
                return
            last_error = f"HTTP {status}"
        except OSError as exc:
            # urllib URLError and connection resets are both OSError descendants.
            # Docker can accept a socket before Uvicorn is ready to answer.
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts - 1:
            time.sleep(delay)
    raise AssertionError(
        f"Demo API did not become healthy at {base_url} "
        f"after {attempts} attempts (last: {last_error})"
    )


def main() -> None:
    wait_for_health()

    status, content_type, html = fetch(Request(BASE_URL + "/"))
    assert status == 200
    assert "text/html" in content_type, content_type
    assert b"Churn Lab" in html
    assert b"/assets/demo.js" in html and b"/assets/demo.css" in html
    assert b"/assets/dashboard-utils.js" in html
    assert b'id="batch-contract-filter"' in html
    assert b'role="tablist"' in html
    assert b'id="batch-cancel"' in html
    assert b'id="batch-no-matches"' in html
    assert b'id="batch-jump-results"' in html
    helper_status, _, helper_script = fetch(Request(BASE_URL + "/assets/dashboard-utils.js"))
    assert helper_status == 200
    assert b"csvExport" in helper_script
    js_status, _, script = fetch(Request(BASE_URL + "/assets/demo.js"))
    assert js_status == 200
    assert b'fetch("/predict"' in script

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
    # A real, vectorized multi-customer request using the same trained artifact.
    stable = {
        **SAMPLE, "Contract": "Two year", "tenure": 60,
        "MonthlyCharges": 50.0, "TotalCharges": 3000.0,
    }
    bulk_request = Request(
        BASE_URL + "/predict/batch",
        data=json.dumps({"customers": [SAMPLE, stable]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    bulk_status, _, bulk_body = fetch(bulk_request)
    assert bulk_status == 200, bulk_body
    bulk = json.loads(bulk_body)
    assert len(bulk["predictions"]) == 2
    assert bulk["summary"]["total_customers"] == 2
    assert 0 <= bulk["summary"]["average_churn_probability"] <= 1
    assert all(0 <= item["churn_probability"] <= 1 for item in bulk["predictions"])
    print(f"Batch demo is operational: {bulk['summary']}")

    # Production demo must never expose the experimental tenant prototype.
    disabled = Request(
        BASE_URL + "/saas/v1/tenants/aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa/me"
    )
    try:
        fetch(disabled)
        raise AssertionError("The disabled SaaS prototype unexpectedly responded 200")
    except HTTPError as error:
        assert error.code == 404, f"Unexpected prototype route status: {error.code}"
        assert error.headers.get("cache-control") == "no-store"

    # The newer OIDC/Postgres staging paths are also disabled on public demo.
    oidc_off = Request(
        BASE_URL + "/saas/v2/tenants/aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa/me"
    )
    try:
        fetch(oidc_off)
        raise AssertionError("The OIDC staging integration unexpectedly responded 200")
    except HTTPError as error:
        assert error.code == 404, f"Unexpected OIDC staging status: {error.code}"
        assert error.headers.get("cache-control") == "no-store"

    # Browser SSO and organization selector are staging-only and must stay off
    # in the public Docker image even without optional identity dependencies.
    for endpoint in (
        "/saas/v3/login",
        "/saas/v3/auth/start",
        "/saas/v3/organizations",
    ):
        try:
            fetch(Request(BASE_URL + endpoint))
            raise AssertionError(f"Private SSO endpoint unexpectedly available: {endpoint}")
        except HTTPError as error:
            assert error.code == 404, (endpoint, error.code)
            assert error.headers.get("cache-control") == "no-store"

    # v1.8 tenant management must be inaccessible in public portfolio Docker.
    import uuid
    tenant = str(uuid.uuid4())
    for method, path, payload in (
        ("POST", "/saas/v3/organizations", {"name": "test"}),
        ("POST", "/saas/v3/invitations/accept", {"code": "A" * 43}),
        ("POST", f"/saas/v3/organizations/{tenant}/invitations", {"role": "viewer"}),
    ):
        request = Request(
            BASE_URL + path,
            data=json.dumps(payload).encode("utf-8"),
            method=method,
            headers={"Content-Type": "application/json", "Origin": BASE_URL},
        )
        try:
            fetch(request)
            raise AssertionError(f"Disabled staging management exposed: {path}")
        except HTTPError as error:
            assert error.code == 404, (path, error.code)
            assert error.headers.get("cache-control") == "no-store"

    print(f"Self-contained demo is operational: {prediction}")


if __name__ == "__main__":
    main()
