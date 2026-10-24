"""Checks for the portfolio UI served by the existing FastAPI application."""

from fastapi.testclient import TestClient

from churn_mlops.api.main import app


def test_root_serves_interactive_spanish_demo() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert '<html lang="es">' in html
    assert 'id="prediction-form"' in html
    assert 'id="result-data"' in html
    assert '<script src="/assets/demo.js" defer></script>' in html
    assert '<link rel="stylesheet" href="/assets/demo.css">' in html
    assert "decision_threshold" in TestClient(app).get("/assets/demo.js").text


def test_demo_includes_all_required_churn_features() -> None:
    client = TestClient(app)
    html = client.get("/").text + "\n" + client.get("/assets/demo.js").text
    fields = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ]
    for field in fields:
        # Inputs/selects are authored directly or built from the static field list.
        assert f'name="{field}"' in html or f'["{field}",' in html


def test_swagger_docs_remains_accessible() -> None:
    response = TestClient(app).get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_static_asset_security_policy() -> None:
    client = TestClient(app)
    script = client.get("/assets/demo.js")
    stylesheet = client.get("/assets/demo.css")
    assert script.status_code == stylesheet.status_code == 200
    assert 'fetch("/predict"' in script.text
    assert "Content-Security-Policy".lower() in script.headers
    assert "script-src 'self'" in script.headers["Content-Security-Policy"]


def test_batch_dashboard_contains_import_controls_and_metrics() -> None:
    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    for control in (
        'id="tab-batch"', 'id="batch-view"', 'id="batch-file"',
        'id="batch-submit"', 'id="batch-chart"', 'id="batch-table"',
        'id="batch-privacy"',
    ):
        assert control in page.text

    script = client.get("/assets/demo.js")
    assert script.status_code == 200
    assert 'fetch("/predict/batch"' in script.text
    assert "batchMaxRows = 200" in script.text
    assert "customersFromCSV" in script.text

    assert "textContent" in script.text


def test_segment_analytics_filters_and_export_are_self_hosted() -> None:
    client = TestClient(app)
    page = client.get("/")
    for identifier in (
        "batch-segments-panel", "contract-segments", "batch-risk-filter",
        "batch-contract-filter", "batch-sort", "batch-export",
        "batch-match-count",
    ):
        assert f'id="{identifier}"' in page.text

    scripts = page.text
    assert scripts.index('src="/assets/dashboard-utils.js"') < scripts.index(
        'src="/assets/demo.js"'
    )
    helper = client.get("/assets/dashboard-utils.js")
    assert helper.status_code == 200
    assert "csvExport" in helper.text
    assert "selectRows" in helper.text
    assert "script-src 'self'" in helper.headers["content-security-policy"]
    assert "unsafe-inline" not in helper.headers["content-security-policy"]


def test_guided_dashboard_keyboard_semantics_and_responsive_feedback() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    for fragment in (
        'class="skip-link"',
        'role="tablist"',
        'role="tab" id="tab-single"',
        'role="tab" id="tab-batch"',
        'role="tabpanel" aria-labelledby="tab-single"',
        'role="tabpanel" aria-labelledby="tab-batch"',
        'class="workflow-steps"',
        'id="batch-cancel"',
        'id="batch-status" role="status"',
        'id="batch-jump-results"',
        'id="batch-clear-filters"',
        'id="batch-no-matches" role="status"',
        'id="batch-results-heading" tabindex="-1"',
    ):
        assert fragment in html

    script = client.get("/assets/demo.js")
    assert script.status_code == 200
    assert "AbortController" in script.text
    assert "fileSelectionGeneration" in script.text
    assert "ArrowRight" in script.text
    assert "batch-no-matches" in script.text

    css = client.get("/assets/demo.css")
    assert css.status_code == 200
    assert "max-width:620px" in css.text
    assert "prefers-reduced-motion:reduce" in css.text
    assert ".cancel-analysis" in css.text
    assert "unsafe-inline" not in css.headers["content-security-policy"]
