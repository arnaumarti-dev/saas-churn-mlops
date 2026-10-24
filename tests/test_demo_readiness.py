"""Regression checks for readiness probes when the demo container starts."""
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_demo_http.py"
spec = importlib.util.spec_from_file_location("check_demo_http", MODULE_PATH)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_health_probe_retries_connection_reset(monkeypatch):
    calls = []

    def fake_fetch(request):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise ConnectionResetError(104, "Connection reset by peer")
        return 200, "application/json", b'{"status":"ok"}'

    monkeypatch.setattr(probe, "fetch", fake_fetch)
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)
    probe.wait_for_health("http://127.0.0.1:8000", attempts=3, delay=0)
    assert calls == ["http://127.0.0.1:8000/health"] * 2


def test_health_probe_exhaustion_reports_last_error(monkeypatch):
    def always_refuse(request):
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr(probe, "fetch", always_refuse)
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)
    with pytest.raises(AssertionError, match="after 2 attempts.*ConnectionRefusedError"):
        probe.wait_for_health("http://127.0.0.1:8000", attempts=2, delay=0)


def test_invalid_retry_count():
    with pytest.raises(ValueError, match="attempts must be positive"):
        probe.wait_for_health(attempts=0)
