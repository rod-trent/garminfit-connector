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
    Routes requests from (after Starlette Mount("/mcp", ...) strips "/mcp"):

      /{access_token}           → FastMCP Streamable HTTP (MCP 2025-03)
      /{access_token}/sse       → FastMCP SSE endpoint (legacy)
      /{access_token}/messages/ → FastMCP SSE messages (legacy)

    Extracts {access_token} from the URL, sets it in user_access_token_var,
    and dispatches to the appropriate handler.
    """

    SSE_SUBPATHS = {"/sse", "/messages/", "/messages"}

    def __init__(self):
        self._sse_app = None     # FastMCP SSE transport (legacy) — lazy-init

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            # Lifespan events: the session manager is managed by the outer
            # lifespan context; nothing to do here.
            return

        full_path: str = scope.get("path", "")
        method: str = scope.get("method", "")

        # Starlette's Mount updates scope["root_path"] but does NOT strip the
        # mount prefix from scope["path"].  We must strip it ourselves so the
        # router sees /{token} instead of /mcp/{token}.
        root_path: str = scope.get("root_path", "")
        if root_path and full_path.startswith(root_path):
            path = full_path[len(root_path):]
            if not path:
                path = "/"
        else:
            path = full_path

        if not path.startswith("/"):
            await self._not_found(scope, receive, send)
            return

        rest = path[1:]   # strip leading "/" -> "{token}" or "{token}/sse"

        if not rest:
            await self._not_found(scope, receive, send)
            return

        slash_idx = rest.find("/")

        if slash_idx == -1:
            # No subpath -- /{token} -> Streamable HTTP
            access_token = rest
            transport = "streamable"
            internal_path = None
        else:
            # Has subpath -- SSE legacy routing
            access_token = rest[:slash_idx]
            sub_path = rest[slash_idx:]   # "/sse", "/messages/", etc.
            if sub_path not in self.SSE_SUBPATHS:
                await self._not_found(scope, receive, send)
                return
            transport = "sse"
            internal_path = sub_path

        if not access_token:
            await self._not_found(scope, receive, send)
            return

        # Log incoming headers for diagnostics
        headers = dict(scope.get("headers", []))
        accept_hdr = headers.get(b"accept", b"").decode()
        content_type_hdr = headers.get(b"content-type", b"").decode()
        session_id_hdr = headers.get(b"mcp-session-id", b"").decode()
        log.warning(
            "[MCP] %s %s (local=%s) -> transport=%s token=%s... "
            "accept=%r content-type=%r session-id=%s",
            method, full_path, path, transport, access_token[:8],
            accept_hdr, content_type_hdr,
            (session_id_hdr[:8] + "...") if session_id_hdr else "(none)",
        )

        # Set the ContextVar so MCP tools can read the user's identity
        token_ctx = user_access_token_var.set(access_token)

        try:
            if transport == "streamable":
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

                # Call the session manager directly -- this avoids the nested-
                # lifespan problem described in the module docstring.
                await mcp.session_manager.handle_request(scope, receive, logging_send)

            else:
                # SSE legacy: lazy-init the sse_app and rewrite the path.
                if self._sse_app is None:
                    self._sse_app = mcp.sse_app()

                new_scope = dict(scope)
                new_scope["path"] = internal_path
                new_scope["raw_path"] = internal_path.encode()
                # Include access_token in root_path so FastMCP advertises the
                # correct messages URL: /mcp/{token}/messages/?session_id=...
                new_scope["root_path"] = (
                    scope.get("root_path", "").rstrip("/") + f"/{access_token}"
                )
                await self._sse_app(new_scope, receive, send)

        except Exception as e:
            log.exception(
                "[MCP] EXCEPTION transport=%s method=%s path=%s token=%s... %s: %s",
                transport, method, full_path, access_token[:8], type(e).__name__, e,
            )
            # Return 500 JSON to the client instead of crashing the ASGI connection
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


app = Starlette(
    lifespan=lifespan,
    routes=[
        # Web pages + API endpoints
        *setup_routes,

        # MCP connector -- everything under /mcp/ goes to GarminMCPRouter
        Mount("/mcp", app=_mcp_router),
    ],
)
