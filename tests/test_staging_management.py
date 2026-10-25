"""Staging-only organization management integration: real PostgreSQL 16 + RLS."""
import hashlib
import os
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("jwt")
from cryptography.fernet import Fernet

from churn_mlops.api.main import app
from churn_mlops.api import staging_login as login


ISSUER = "https://login.example.org/"


@pytest.fixture()
def config(monkeypatch):
    values = {
        "CHURN_SAAS_MANAGEMENT_ENABLED": "1",
        "CHURN_SAAS_LOGIN_ENABLED": "1",
        "CHURN_SAAS_OIDC_ENABLED": "1",
        "CHURN_SAAS_OIDC_ISSUER": ISSUER,
        "CHURN_SAAS_OIDC_AUDIENCE": "churn-mgmt-api",
        "CHURN_SAAS_OIDC_JWKS_URL": "https://login.example.org/.well-known/jwks.json",
        "CHURN_SAAS_OIDC_CLIENT_ID": "churn-mgmt-browser",
        "CHURN_SAAS_AUTHORIZATION_URL": "https://login.example.org/authorize",
        "CHURN_SAAS_TOKEN_URL": "https://login.example.org/oauth/token",
        "CHURN_SAAS_REDIRECT_URI": "https://testserver/saas/v3/auth/callback",
        "CHURN_SAAS_DATABASE_URL": os.getenv("CHURN_TEST_DB_APP_URL", "postgresql://unused.invalid/test"),
        "CHURN_SAAS_SESSION_FERNET_KEY": Fernet.generate_key().decode(),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


def browser(subject):
    # Separate synthetic clients by IP: the ASGI production IP throttle is
    # global to the app and should not mistake distinct test identities for
    # one abusive client. Each synthetic browser stays under its own quota.
    digest = hashlib.sha256(subject.encode()).digest()
    source_ip = f"10.{digest[0]}.{digest[1]}.{digest[2]}"
    client = TestClient(
        app, base_url="https://testserver", follow_redirects=False,
        client=(source_ip, 51000),
    )
    cfg = login._login_config()
    client.cookies.set(
        login.SESSION_COOKIE,
        login._seal(cfg, {"iss": ISSUER, "sub": subject,
                          "exp": int(time.time()) + 600}),
        path="/",
    )
    return client


def post(client, path, body):
    return client.post(path, json=body, headers={"Origin": "https://testserver"})


def patch(client, path, body):
    return client.patch(path, json=body, headers={"Origin": "https://testserver"})


def delete(client, path):
    return client.delete(path, headers={"Origin": "https://testserver"})


def test_management_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CHURN_SAAS_MANAGEMENT_ENABLED", raising=False)
    with TestClient(app, base_url="https://testserver") as client:
        checks = [
            ("post", "/saas/v3/organizations", {"name": "Trial"}),
            ("post", "/saas/v3/invitations/accept", {"code": "a" * 43}),
            ("post", f"/saas/v3/organizations/{uuid4()}/invitations", {"role": "viewer"}),
            ("get", f"/saas/v3/organizations/{uuid4()}/members", None),
            ("patch", f"/saas/v3/organizations/{uuid4()}/members/example", {"role": "owner"}),
            ("delete", f"/saas/v3/organizations/{uuid4()}/members/example", None),
        ]
        for method, path, body in checks:
            kwargs = ({"json": body} if body is not None else {})
            response = getattr(client, method)(path, **kwargs)
            assert response.status_code == 404, (method, path, response.text)
            assert response.headers.get("cache-control") == "no-store"


def test_require_cookie_and_strict_origin(config):
    with TestClient(app, base_url="https://testserver") as guest:
        assert post(guest, "/saas/v3/organizations", {"name": "Trial"}).status_code == 401
    with browser("actor-v18") as user:
        assert user.post("/saas/v3/organizations", json={"name": "Trial"}).status_code == 403
        assert user.post("/saas/v3/organizations", json={"name": "Trial"},
                         headers={"Origin": "https://evil.example"}).status_code == 403
        assert user.post("/saas/v3/organizations", data="name=Trial",
                         headers={"Origin": "https://testserver",
                                  "Content-Type": "application/x-www-form-urlencoded"}).status_code == 415


@pytest.mark.skipif(not os.getenv("CHURN_TEST_DB_ADMIN_URL"), reason="Ephemeral PostgreSQL required")
def test_tenant_creation_invites_roles_revocation_last_owner_and_rls(config):
    owner_name, second_name, outsider_name = (
        "owner-" + str(uuid4()), "member-" + str(uuid4()), "other-" + str(uuid4())
    )
    with browser(owner_name) as owner, browser(second_name) as member, browser(outsider_name) as outsider:
        created = post(owner, "/saas/v3/organizations", {"name": "  Prueba SaaS  "})
        assert created.status_code == 201, created.text
        tenant = created.json()["tenant_id"]
        assert created.json()["name"] == "Prueba SaaS"
        tenant_path = f"/saas/v3/organizations/{tenant}"
        assert [o["name"] for o in login._my_orgs(login.Principal(ISSUER, owner_name))
                if o["tenant_id"] == tenant] == ["Prueba SaaS"]

        # The last active owner must not leave or downgrade themselves.
        assert patch(owner, f"{tenant_path}/members/{owner_name}",
                     {"role": "viewer"}).status_code == 403
        assert delete(owner, f"{tenant_path}/members/{owner_name}").status_code == 403
        assert outsider.get(tenant_path + "/members").status_code == 403
        assert post(outsider, tenant_path + "/invitations", {"role": "analyst"}).status_code == 403

        issue = post(owner, tenant_path + "/invitations", {"role": "viewer"})
        assert issue.status_code == 201, issue.text
        code = issue.json()["code"]
        assert len(code) == 43
        digest = hashlib.sha256(code.encode()).hexdigest()
        with psycopg.connect(os.environ["CHURN_TEST_DB_ADMIN_URL"]) as db:
            row = db.execute("SELECT token_hash, role FROM saas.invitations WHERE token_hash=%s",
                             (digest,)).fetchone()
            assert row == (digest, "viewer")
            assert code not in str(row)
        assert post(outsider, "/saas/v3/invitations/accept", {"code": code + "x"}).status_code == 422
        accepted = post(member, "/saas/v3/invitations/accept", {"code": code})
        assert accepted.status_code == 200, accepted.text
        assert accepted.json()["tenant_id"] == tenant
        assert post(outsider, "/saas/v3/invitations/accept", {"code": code}).status_code == 403

        # Only owners can view or edit memberships; viewers cannot invite.
        assert member.get(tenant_path + "/members").status_code == 403
        assert post(member, tenant_path + "/invitations", {"role": "viewer"}).status_code == 403
        listed = owner.get(tenant_path + "/members")
        assert listed.status_code == 200
        assert {x["subject"] for x in listed.json()["members"]} == {owner_name, second_name}
        assert {x["role"] for x in listed.json()["members"]} == {"owner", "viewer"}

        # Promote a second owner, then safely remove the original.
        assert patch(owner, f"{tenant_path}/members/{second_name}",
                     {"role": "owner"}).status_code == 200
        assert patch(owner, f"{tenant_path}/members/{owner_name}",
                     {"role": "analyst"}).status_code == 200
        assert post(owner, tenant_path + "/invitations", {"role": "viewer"}).status_code == 403
        assert delete(member, f"{tenant_path}/members/{owner_name}").status_code == 200
        assert owner.get(tenant_path + "/members").status_code == 403
        assert delete(member, f"{tenant_path}/members/{second_name}").status_code == 403

        # Membership can be reactivated only by accepting a NEW unexpired invite.
        second_code = post(member, tenant_path + "/invitations",
                           {"role": "analyst"}).json()["code"]
        assert post(owner, "/saas/v3/invitations/accept",
                    {"code": second_code}).status_code == 200
        assert patch(owner, f"{tenant_path}/members/{second_name}",
                     {"role": "viewer"}).status_code == 403

        # No app DB role may bypass table grants directly.
        app_dsn = os.environ["CHURN_TEST_DB_APP_URL"]
        with psycopg.connect(app_dsn) as db:
            assert db.execute("SELECT count(*) FROM saas.memberships").fetchone()[0] == 0
        with psycopg.connect(app_dsn) as db:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                db.execute("SELECT * FROM saas.invitations")
        with psycopg.connect(app_dsn) as db:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                db.execute("UPDATE saas.memberships SET role='owner'")


@pytest.mark.skipif(not os.getenv("CHURN_TEST_DB_ADMIN_URL"), reason="Ephemeral PostgreSQL required")
def test_expired_invite_and_no_cross_tenant_owner_access(config):
    person_a, person_b = "a-" + str(uuid4()), "b-" + str(uuid4())
    with browser(person_a) as a, browser(person_b) as b:
        one = post(a, "/saas/v3/organizations", {"name": "A"}).json()["tenant_id"]
        two = post(b, "/saas/v3/organizations", {"name": "B"}).json()["tenant_id"]
        assert a.get(f"/saas/v3/organizations/{two}/members").status_code == 403
        assert patch(b, f"/saas/v3/organizations/{one}/members/{person_a}",
                     {"role": "viewer"}).status_code == 403
        code = post(a, f"/saas/v3/organizations/{one}/invitations",
                    {"role": "analyst"}).json()["code"]
        digest = hashlib.sha256(code.encode()).hexdigest()
        with psycopg.connect(os.environ["CHURN_TEST_DB_ADMIN_URL"]) as db:
            db.execute("UPDATE saas.invitations SET expires_at=now() - interval '1 minute' "
                       "WHERE token_hash=%s", (digest,))
        assert post(b, "/saas/v3/invitations/accept", {"code": code}).status_code == 403
        assert not any(x["tenant_id"] == one for x in login._my_orgs(login.Principal(ISSUER, person_b)))


@pytest.mark.skipif(not os.getenv("CHURN_TEST_DB_ADMIN_URL"), reason="Ephemeral PostgreSQL required")
def test_fictional_alias_is_self_only_unique_per_tenant_and_revocable(config):
    owner_id, viewer_id, outsider_id = (str(uuid4()) for _ in range(3))
    with browser(owner_id) as owner, browser(viewer_id) as viewer, browser(outsider_id) as outsider:
        company = post(owner, "/saas/v3/organizations", {"name": "AliasCo"}).json()["tenant_id"]
        another = post(owner, "/saas/v3/organizations", {"name": "SecondCo"}).json()["tenant_id"]
        alias_url = f"/saas/v3/organizations/{company}/me/alias"
        headers = {"Origin": "https://testserver"}

        # Every member can change only their own fictional alias.
        assert owner.put(alias_url, json={"alias": "Ana Demo"}).status_code == 403
        assert owner.put(alias_url, json={"alias": "A"}, headers=headers).status_code == 422
        assert owner.put(alias_url, json={"alias": "Ana@personal.com"}, headers=headers).status_code == 422
        assert outsider.put(alias_url, json={"alias": "Intruso"}, headers=headers).status_code == 403
        saved = owner.put(alias_url, json={"alias": "  Ana Demo  "}, headers=headers)
        assert saved.status_code == 200, saved.text
        assert saved.json()["alias"] == "Ana Demo"
        my_orgs = owner.get("/saas/v3/organizations/list").json()["organizations"]
        assert next(o for o in my_orgs if o["tenant_id"] == company)["alias"] == "Ana Demo"
        assert next(o for o in my_orgs if o["tenant_id"] == another)["alias"] is None

        token = post(owner, f"/saas/v3/organizations/{company}/invitations",
                     {"role": "viewer"}).json()["code"]
        assert post(viewer, "/saas/v3/invitations/accept", {"code": token}).status_code == 200
        viewer_saved = viewer.put(alias_url, json={"alias": "Luis Demo"}, headers=headers)
        assert viewer_saved.status_code == 200, viewer_saved.text
        assert viewer.get(f"/saas/v3/organizations/{company}/members").status_code == 403
        listing = owner.get(f"/saas/v3/organizations/{company}/members")
        assert listing.status_code == 200
        members = {item["subject"]: item for item in listing.json()["members"]}
        assert members[owner_id]["alias"] == "Ana Demo"
        assert members[viewer_id]["alias"] == "Luis Demo"

        # Duplicate aliases cannot trick owners into editing a lookalike account.
        duplicate = viewer.put(alias_url, json={"alias": "ana demo"}, headers=headers)
        assert duplicate.status_code == 409, duplicate.text
        assert outsider.get(f"/saas/v3/organizations/{company}/members").status_code == 403
        assert owner.get(f"/saas/v3/organizations/{another}/members").json()["members"][0]["alias"] is None

        # Revoked members cannot rename themselves or access membership details.
        assert delete(owner, f"/saas/v3/organizations/{company}/members/{viewer_id}").status_code == 200
        assert viewer.put(alias_url, json={"alias": "Luis V2"}, headers=headers).status_code == 403
        assert viewer.get("/saas/v3/organizations/list").json()["organizations"] == []

        # App credentials have no direct UPDATE permission, including aliases.
        with psycopg.connect(os.environ["CHURN_TEST_DB_APP_URL"]) as db:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                db.execute("UPDATE saas.memberships SET display_alias = 'Forged'")
