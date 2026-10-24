"""ASGI request-size, abuse-control and response-header protections.

These in-process rate limits are best-effort on one worker. A production
service would need shared storage or an edge gateway for globally enforced
limits across replicas and restarts.
"""
from collections import deque
import re
from threading import Lock
from time import monotonic

from starlette.responses import JSONResponse


MAX_PREDICT_BODY = 16 * 1024
MAX_BATCH_BODY = 256 * 1024
PREDICTIONS_PER_MINUTE = 30
MAX_RATE_LIMIT_IDENTITIES = 4096
TENANT_BATCH_PATH = re.compile(r"^/saas/v1/tenants/[^/]+/predict/batch$")

CSP = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "connect-src 'self'; "
    "img-src 'self'; "
    "font-src 'none'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'; "
    "object-src 'none'"
)

# FastAPI's built-in Swagger UI loads scripts/styles from a CDN and uses
# inline initialization. Restrict that exception to /docs and /redoc only.
DOCS_CSP = (
    "default-src 'none'; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "connect-src 'self'; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' https://cdn.jsdelivr.net; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
    "object-src 'none'"
)

SECURITY_HEADERS = [
    (b"content-security-policy", CSP.encode("ascii")),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    (b"strict-transport-security", b"max-age=31536000"),
]


class RequestRateLimiter:
    """Fixed sliding-window rate limit per client IP, local to one process."""

    def __init__(self, limit: int = PREDICTIONS_PER_MINUTE, period: float = 60.0):
        self.limit = limit
        self.period = period
        self._timestamps: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, client: str) -> bool:
        now = monotonic()
        with self._lock:
            timestamps = self._timestamps.get(client)
            if timestamps is None:
                if len(self._timestamps) >= MAX_RATE_LIMIT_IDENTITIES:
                    # Evict expired inactive entries. If still at capacity,
                    # reject newcomers rather than letting memory grow forever.
                    for key, values in list(self._timestamps.items()):
                        if not values or now - values[-1] >= self.period:
                            del self._timestamps[key]
                    if len(self._timestamps) >= MAX_RATE_LIMIT_IDENTITIES:
                        return False
                timestamps = deque()
                self._timestamps[client] = timestamps
            while timestamps and now - timestamps[0] >= self.period:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                return False
            timestamps.append(now)
            return True


class SecurityMiddleware:
    """Pure ASGI middleware to cap upload size *before* passing to Pydantic."""

    def __init__(self, app):
        self.app = app
        self.limiter = RequestRateLimiter()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def secure_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {key.lower() for key, _ in headers}
                for key, value in SECURITY_HEADERS:
                    if key not in existing:
                        if key == b"content-security-policy" and scope["path"] in {"/docs", "/redoc"}:
                            value = DOCS_CSP.encode("ascii")
                        headers.append((key, value))
                if scope["path"] in {"/predict", "/predict/batch"} or scope["path"].startswith("/saas/v1/"):
                    headers.append((b"cache-control", b"no-store"))
                message = {**message, "headers": headers}
            await send(message)

        async def reject(code, detail, extra_headers=None):
            await JSONResponse(
                {"detail": detail}, status_code=code, headers=extra_headers
            )(scope, receive, secure_send)

        is_tenant_batch = bool(TENANT_BATCH_PATH.fullmatch(scope.get("path", "")))
        if (scope.get("path") in {"/predict", "/predict/batch"} or is_tenant_batch) and scope.get("method") == "POST":
            body_limit = MAX_BATCH_BODY if (scope["path"] == "/predict/batch" or is_tenant_batch) else MAX_PREDICT_BODY
            ip = (scope.get("client") or ("unknown", 0))[0]
            if not self.limiter.allow(ip):
                await reject(429, "Too many predictions; try again shortly.", {"Retry-After": "60"})
                return

            headers = dict(scope.get("headers", []))
            media_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip().lower()
            if media_type != b"application/json":
                await reject(415, "Only application/json requests are accepted.")
                return

            try:
                declared_length = int(headers.get(b"content-length", b"0"))
            except ValueError:
                await reject(400, "Invalid Content-Length.")
                return

            if declared_length > body_limit:
                await reject(413, "Prediction request too large.")
                return

            # Read bounded chunks even when transfer-encoding is chunked.
            chunks = []
            total = 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    continue
                body = message.get("body", b"")
                total += len(body)
                if total > body_limit:
                    await reject(413, "Prediction request too large.")
                    return
                chunks.append(body)
                if not message.get("more_body", False):
                    break

            payload = b"".join(chunks)
            sent = False

            async def replay():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": payload, "more_body": False}
                return await receive()

            await self.app(scope, replay, secure_send)
            return

        await self.app(scope, receive, secure_send)
