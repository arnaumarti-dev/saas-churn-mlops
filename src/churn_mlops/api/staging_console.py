"""Default-off, authenticated browser console for private synthetic SaaS staging.

Unlike public /assets this router serves the HTML, CSS and JS only after
the same OIDC session + feature flags required by management operations.
It never embeds issuer/subject, secrets, organization names or invite tokens
into HTML; the browser retrieves current membership through the existing DB.
"""
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from churn_mlops.api.oidc_postgres import Principal
from churn_mlops.api.staging_login import _my_orgs
from churn_mlops.api.staging_management import _v18_principal

router = APIRouter(prefix="/saas/v3")
PRIVATE_ASSETS = Path(__file__).resolve().parent / "private_assets"


@router.get("/console", include_in_schema=False)
def console(_: Principal = Depends(_v18_principal)):
    return FileResponse(
        PRIVATE_ASSETS / "console.html", media_type="text/html",
    )


@router.get("/console.js", include_in_schema=False)
def console_script(_: Principal = Depends(_v18_principal)):
    return FileResponse(
        PRIVATE_ASSETS / "console.js", media_type="application/javascript",
    )


@router.get("/console.css", include_in_schema=False)
def console_styles(_: Principal = Depends(_v18_principal)):
    return FileResponse(
        PRIVATE_ASSETS / "console.css", media_type="text/css",
    )


@router.get("/organizations/list")
def list_organizations(principal: Principal = Depends(_v18_principal)):
    # Postgres enforces issuer/subject membership and tenant-specific names;
    # never fetch all organizations or accept a claimed caller-supplied tenant.
    return {"organizations": _my_orgs(principal)}
