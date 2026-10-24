"""Regression tests for protections around the public prediction route."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from churn_mlops.api.main import app
from churn_mlops.api.security import MAX_PREDICT_BODY, SecurityMiddleware


def test_security_headers_cover_dashboard_and_api() -> None:
    client = TestClient(app)
    for path in ("/", "/health", "/docs", "/assets/demo.js"):
        response = client.get(path)
        assert response.status_code == 200
        assert "script-src 'self'" in response.headers["content-security-policy"]
        assert "unsafe-inline" not in response.headers["content-security-policy"]
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"


def isolated_app() -> FastAPI:
    """Avoid modifying the main application's real model and rate-limit state."""
    demo = FastAPI()
    demo.add_middleware(SecurityMiddleware)

    @demo.post("/predict")
    def dummy_prediction() -> dict:
        return {"churn_probability": 0.9}

    return demo


def test_rejects_oversized_post_body() -> None:
    client = TestClient(isolated_app())
    response = client.post("/predict", json={"padding": "x" * MAX_PREDICT_BODY})
    assert response.status_code == 413
    assert response.headers["cache-control"] == "no-store"


def test_rejects_unsupported_media_type() -> None:
    client = TestClient(isolated_app())
    response = client.post("/predict", content="abc", headers={"Content-Type": "text/plain"})
    assert response.status_code == 415


def test_rejects_oversized_stream_without_content_length() -> None:
    client = TestClient(isolated_app())

    def stream():
        for _ in range(20):
            yield b"x" * 1024

    response = client.post(
        "/predict", content=stream(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_in_process_rate_limit_responds_with_retry_after() -> None:
    client = TestClient(isolated_app())
    for _ in range(30):
        response = client.post("/predict", json={"value": 1})
        assert response.status_code == 200

    response = client.post("/predict", json={"value": 1})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert response.headers["cache-control"] == "no-store"
