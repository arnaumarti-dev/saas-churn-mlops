"""Real Chromium flow against TLS FastAPI + ephemeral PostgreSQL 16.

This does NOT test a hosted OIDC provider. Only synthetic identities are
represented by short-lived Fernet browser sessions generated for CI. It never
contacts a public service, checks in credentials or uses customer records.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
from uuid import uuid4

import pytest

pytest.importorskip("psycopg")
playwright = pytest.importorskip("playwright.sync_api")
from cryptography.fernet import Fernet

from playwright.sync_api import expect, sync_playwright


pytestmark = pytest.mark.skipif(
    not os.environ.get("CHURN_TEST_DB_APP_URL"),
    reason="Requires the CI-only ephemeral PostgreSQL app role",
)
ISSUER = "https://login.example.org/"
COOKIE_NAME = "__Host-churn_saas_session"


@pytest.fixture(scope="module")
def staging(tmp_path_factory):
    """Start the real service on ephemeral TLS; never disable secure cookies."""
    root = Path(__file__).resolve().parents[1]
    folder = tmp_path_factory.mktemp("private-browser-staging")
    key = Fernet.generate_key().decode("ascii")
    # Browser callback validation deliberately requires default HTTPS port 443.
    # CI enables unprivileged binding in its disposable Linux runner only.
    port = 443
    url = "https://localhost"
    cert, private_key = folder / "test-cert.pem", folder / "test-key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256",
            "-nodes", "-days", "1", "-keyout", str(private_key),
            "-out", str(cert), "-subj", "/CN=localhost",
            "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "CHURN_SAAS_OIDC_ENABLED": "1",
        "CHURN_SAAS_LOGIN_ENABLED": "1",
        "CHURN_SAAS_MANAGEMENT_ENABLED": "1",
        "CHURN_SAAS_OIDC_ISSUER": ISSUER,
        "CHURN_SAAS_OIDC_AUDIENCE": "churn-browser-ci-api",
        "CHURN_SAAS_OIDC_CLIENT_ID": "churn-browser-ci-client",
        "CHURN_SAAS_OIDC_JWKS_URL": "https://login.example.org/.well-known/jwks.json",
        "CHURN_SAAS_AUTHORIZATION_URL": "https://login.example.org/authorize",
        "CHURN_SAAS_TOKEN_URL": "https://login.example.org/oauth/token",
        "CHURN_SAAS_REDIRECT_URI": url + "/saas/v3/auth/callback",
        "CHURN_SAAS_DATABASE_URL": os.environ["CHURN_TEST_DB_APP_URL"],
        "CHURN_SAAS_SESSION_FERNET_KEY": key,
    }
    # No logs written to GitHub artifacts and no user-provided credentials.
    with (folder / "server.log").open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "churn_mlops.api.main:app",
                "--host", "127.0.0.1", "--port", str(port),
                "--ssl-certfile", str(cert), "--ssl-keyfile", str(private_key),
            ], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            tls = ssl._create_unverified_context()  # ephemeral self-signed CI cert only
            for _ in range(75):
                if process.poll() is not None:
                    raise RuntimeError("Private TLS test server terminated early")
                try:
                    with urlopen(url + "/health", timeout=2, context=tls) as response:
                        if response.status == 200:
                            break
                except (OSError, URLError):
                    pass
                time.sleep(0.4)
            else:
                raise RuntimeError("Private TLS test server did not start")
            yield {"url": url, "key": key}
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture(scope="module")
def chromium():
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            browser.close()


def signed_context(browser, staging, subject, viewport):
    context = browser.new_context(
        viewport=viewport, ignore_https_errors=True,
        locale="es-ES", reduced_motion="reduce",
    )
    payload = {
        "iss": ISSUER, "sub": subject,
        "exp": int(time.time()) + 600,
    }
    token = Fernet(staging["key"].encode("ascii")).encrypt(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    context.add_cookies([{
        "name": COOKIE_NAME, "value": token,
        "url": staging["url"] + "/",
        "httpOnly": True, "secure": True, "sameSite": "Lax",
    }])
    return context


@pytest.mark.parametrize("viewport", [
    {"width": 1390, "height": 870},
    {"width": 390, "height": 844},
], ids=["desktop", "mobile"])
def test_real_private_browser_organization_lifecycle(
    chromium, staging, viewport, request
):
    url = staging["url"]
    # Without an actual session neither the console nor its JS/CSS are public.
    guest = chromium.new_context(ignore_https_errors=True)
    try:
        for path in ("/console", "/console.js", "/console.css", "/organizations/list"):
            res = guest.request.get(url + "/saas/v3" + path)
            assert res.status == 401, (path, res.status)
    finally:
        guest.close()

    owner_subject = "owner-" + str(uuid4())
    member_subject = "member-" + str(uuid4())
    owner_context = signed_context(chromium, staging, owner_subject, viewport)
    member_context = signed_context(chromium, staging, member_subject, viewport)
    try:
        owner = owner_context.new_page()
        member = member_context.new_page()
        response = owner.goto(url + "/saas/v3/console", wait_until="networkidle")
        assert response.status == 200
        expect(owner.locator("#feedback")).to_have_text("Panel listo.")
        expect(owner.locator("#org-picker")).to_be_disabled()
        expect(owner.locator("#owner-tools")).to_be_hidden()

        name = "Acme pruebas " + str(uuid4())[:8]
        owner.locator("#org-name").fill(name)
        owner.locator('#create-org button[type="submit"]').click()
        expect(owner.locator("#feedback")).to_have_text("Empresa de pruebas creada.")
        expect(owner.locator("#org-picker")).to_contain_text(name)
        expect(owner.locator("#owner-tools")).to_be_visible()
        expect(owner.locator("#invite-tools")).to_be_visible()
        expect(owner.locator(".member").filter(has_text=owner_subject)).to_be_visible()

        # Capture ONLY synthetic, non-secret data; no invitation code is shown.
        folder = Path("test-results/private-staging-qa")
        folder.mkdir(parents=True, exist_ok=True)
        owner.screenshot(
            path=str(folder / (request.node.callspec.id + ".png")),
            full_page=True,
        )

        # The database, not the UI, forbids the last owner losing ownership.
        owner_row = owner.locator(".member").filter(has_text=owner_subject)
        owner_row.locator("select").select_option("viewer")
        expect(owner.locator("#feedback")).to_have_text(
            "No tienes permisos para realizar esta acción."
        )
        expect(owner_row.locator("select")).to_have_value("owner")

        # Owner issues a one-time invitation code; it never appears in a URL.
        owner.locator("#invite-role").select_option("viewer")
        owner.locator('#issue-invite button[type="submit"]').click()
        expect(owner.locator("#one-time-code")).to_be_visible()
        code = owner.locator("#issued-code").text_content()
        assert len(code) == 43 and "/" not in code and "+" not in code
        owner.locator("#hide-code").click()
        expect(owner.locator("#issued-code")).to_be_empty()

        member.goto(url + "/saas/v3/console", wait_until="networkidle")
        expect(member.locator("#feedback")).to_have_text("Panel listo.")
        member.locator("#invite-input").fill(code)
        member.locator('#redeem-invite button[type="submit"]').click()
        expect(member.locator("#feedback")).to_have_text("Invitación aceptada.")
        expect(member.locator("#org-picker")).to_contain_text(name)
        expect(member.locator("#owner-tools")).to_be_hidden()
        expect(member.locator("#invite-tools")).to_be_hidden()

        # A viewer may see their organization but not the owner's user list.
        org = owner.locator("#org-picker").input_value()
        assert org
        unauthorized = member_context.request.get(
            url + "/saas/v3/organizations/" + org + "/members"
        )
        assert unauthorized.status == 403

        # Owner can promote and revoke a member. UI re-reads actual DB state.
        owner.locator("#refresh-members").click()
        member_row = owner.locator(".member").filter(has_text=member_subject)
        expect(member_row).to_be_visible()
        member_row.locator("select").select_option("analyst")
        expect(owner.locator("#feedback")).to_have_text("Permisos actualizados.")
        expect(owner.locator(".member").filter(has_text=member_subject)
               .locator("select")).to_have_value("analyst")

        owner.once("dialog", lambda dialog: dialog.accept())
        owner.locator(".member").filter(has_text=member_subject).get_by_role(
            "button", name="Revocar"
        ).click()
        expect(owner.locator("#feedback")).to_have_text("Acceso revocado.")
        expect(owner.locator(".member").filter(has_text=member_subject)).to_have_count(0)
        member.locator("#refresh").click()
        expect(member.locator("#org-picker")).to_be_disabled()
        expect(member.locator("#owner-tools")).to_be_hidden()
    finally:
        owner_context.close()
        member_context.close()
