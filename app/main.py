"""
Starlette application entry point.

Key components:
  1. GarminMCPRouter — custom ASGI router that:
       - Parses {access_token} from /mcp/{access_token} (Streamable HTTP)
         and from /mcp/{access_token}/sse + /mcp/{access_token}/messages/ (SSE)
       - Sets user_access_token_var ContextVar so MCP tools can identify the user
       - For Streamable HTTP: calls mcp.session_manager.handle_request() directly
       - For SSE (legacy):    rewrites path and delegates to FastMCP's sse_app

  2. setup_routes — HTML setup/disconnect pages and /api/* JSON endpoints

  3. lifespan — creates DB tables and starts the MCP session manager

The app is served by uvicorn:
  uvicorn app.main:app --host 0.0.0.0 --port $PORT

Transport support:
  - Streamable HTTP (MCP 2025-03): POST/GET /mcp/{token}   ← Claude.ai uses this
  - SSE (legacy):                  GET /mcp/{token}/sse    ← kept for compatibility

Why session_manager is started in the outer lifespan
-----------------------------------------------------
FastMCP's streamable_http_app() returns a Starlette app whose lifespan runs
session_manager.run(). When that inner Starlette is mounted inside the outer one
they share the same ASGI lifespan receive/send channel, which the outer app
consumes first — so the inner lifespan never fires and session_manager._task_group
stays None, causing every request to raise RuntimeError.

The fix: call mcp.streamable_http_app() once (for the side-effect of creating
mcp._session_manager), then run session_manager.run() from the outer lifespan
context manager, and route Streamable HTTP requests directly to
mcp.session_manager.handle_request() — bypassing the now-unused inner Starlette.
"""

import contextlib
import logging
import sys

from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

# ---------------------------------------------------------------------------
# Force stdout to be unbuffered so print() appears immediately in Railway logs
# ---------------------------------------------------------------------------
sys.stdout.reconfigure(line_buffering=True)

log = logging.getLogger("garminfit.mcp")

from app.database import create_tables
from app.mcp_server import mcp, user_access_token_var
from app.setup_routes import setup_routes


# ---------------------------------------------------------------------------
# Custom ASGI router for per-user MCP URLs
# ---------------------------------------------------------------------------

class GarminMCPRouter:
    """
    Handles MCP Streamable HTTP (2025-03-26) requests mounted at /garmin.

    Token identity is carried in the query string, not the URL path:
      POST/GET https://…/garmin/?token={access_token}

    Why query params instead of path segments:
      FastMCP's session manager returns 421 for any request whose path has a
      subpath beyond the mount root (e.g. /garmin/abc → 421).  Moving the
      token to ?token= keeps the effective path at /garmin/ (the mount root)
      so the session manager accepts every request.
    """

    def __init__(self):
        pass

    async def __call__(self, scope, receive, send):
        full_path: str = scope.get("path", "")
        method: str = scope.get("method", "")

        log.warning("[MCP-ROUTER] called type=%s method=%s path=%s", scope["type"], method, full_path)

        if scope["type"] not in ("http", "websocket"):
            return

        # Extract token from query string: ?token={access_token}
        query_string: bytes = scope.get("query_string", b"")
        access_token: str | None = None
        for part in query_string.split(b"&"):
            if part.startswith(b"token="):
                access_token = part[len(b"token="):].decode()
                break

        if not access_token:
            log.warning("[MCP-ROUTER] 404 no ?token= in query string (path=%s qs=%r)", full_path, query_string)
            await self._not_found(scope, receive, send)
            return

        # Log incoming headers for diagnostics
        headers = dict(scope.get("headers", []))
        accept_hdr = headers.get(b"accept", b"").decode()
        content_type_hdr = headers.get(b"content-type", b"").decode()
        session_id_hdr = headers.get(b"mcp-session-id", b"").decode()
        log.warning(
            "[MCP] %s %s token=%s... accept=%r content-type=%r session-id=%s",
            method, full_path, access_token[:8],
            accept_hdr, content_type_hdr,
            (session_id_hdr[:8] + "...") if session_id_hdr else "(none)",
        )

        # Set the ContextVar so MCP tools can read the user's identity
        token_ctx = user_access_token_var.set(access_token)

        try:
            # Intercept send to log the response status
            async def logging_send(event):
                if event.get("type") == "http.response.start":
                    status = event.get("status", "?")
                    resp_headers = {
                        k.decode(): v.decode()
                        for k, v in event.get("headers", [])
                        if isinstance(k, bytes)
                    }
                    log.warning(
                        "[MCP] response status=%s session-id=%s",
                        status,
                        resp_headers.get("mcp-session-id", "(none)")[:16] + "...",
                    )
                await send(event)

            await mcp.session_manager.handle_request(scope, receive, logging_send)

        except Exception as e:
            log.exception(
                "[MCP] EXCEPTION method=%s path=%s token=%s... %s: %s",
                method, full_path, access_token[:8], type(e).__name__, e,
            )
            if scope["type"] == "http":
                try:
                    import json as _json
                    body = _json.dumps({"error": str(e), "type": type(e).__name__}).encode()
                    await send({"type": "http.response.start", "status": 500,
                                "headers": [[b"content-type", b"application/json"]]})
                    await send({"type": "http.response.body", "body": body})
                except Exception:
                    pass  # send already started — nothing we can do
        finally:
            user_access_token_var.reset(token_ctx)

    @staticmethod
    async def _not_found(scope, receive, send):
        if scope["type"] == "http":
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [[b"content-type", b"application/json"]],
            })
            await send({
                "type": "http.response.body",
                "body": b'{"error": "MCP endpoint not found. Check your connector URL."}',
            })


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

_mcp_router = GarminMCPRouter()


# ---------------------------------------------------------------------------
# Raw ASGI middleware: logs every request BEFORE Starlette routing touches it
# ---------------------------------------------------------------------------

class RequestLogMiddleware:
    """Logs the raw path/method of every HTTP request before any routing."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            log.warning(
                "[RAW] %s %s",
                scope.get("method", "?"),
                scope.get("path", "?"),
            )
        await self._app(scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan(app):
    """
    Starlette lifespan context manager.

    1. Creates database tables (idempotent).
    2. Initialises FastMCP's StreamableHTTP session manager and keeps it
       running for the duration of the server's life.
    """
    import os
    log.warning("garminfit-connector starting up")
    log.warning("  DATABASE_URL set: %s", "yes" if os.environ.get("DATABASE_URL") else "NO -- using SQLite fallback")
    log.warning("  TOKEN_ENCRYPTION_KEY set: %s", "yes" if os.environ.get("TOKEN_ENCRYPTION_KEY") else "NO -- will crash on setup")
    log.warning("  APP_BASE_URL: %s", os.environ.get("APP_BASE_URL", "(not set -- using request host)"))

    try:
        await create_tables()
        log.warning("Database tables ready")
    except Exception as e:
        log.warning("Database startup warning: %s", e)
        log.warning("  App will continue -- DB may not be ready yet")

    # mcp.streamable_http_app() creates mcp._session_manager if it doesn't
    # exist yet (idempotent; we discard the returned Starlette app because
    # we call the session manager directly).
    mcp.streamable_http_app()
    log.warning("MCP session manager initialising")

    async with mcp.session_manager.run():
        log.warning("MCP session manager started -- ready to accept connections")
        yield

    log.warning("MCP session manager stopped")


_starlette = Starlette(
    lifespan=lifespan,
    routes=[
        # Web pages + API endpoints
        *setup_routes,

        # Static assets (favicon, etc.)
        Mount("/static", app=StaticFiles(directory="static"), name="static"),

        # MCP connector — all requests to /garmin (and /garmin/) go here.
        # Token is in the query string: /garmin/?token={access_token}
        # Path-based tokens (/garmin/{token}) caused FastMCP to return 421.
        Mount("/garmin", app=_mcp_router),
    ],
)

# Wrap with raw-request logger so we can see every request before routing
app = RequestLogMiddleware(_starlette)
