"""
Starlette application entry point.

Key components:
  1. GarminMCPRouter — custom ASGI router that:
       - Parses {access_token} from /mcp/{access_token}/sse and /mcp/{access_token}/messages/
       - Sets user_access_token_var ContextVar so MCP tools can identify the user
       - Rewrites the path to /sse or /messages/ before delegating to FastMCP's SSE app

  2. setup_routes — HTML setup/disconnect pages and /api/* JSON endpoints

  3. on_startup — creates DB tables if they don't exist

The app is served by uvicorn:
  uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse

from app.database import create_tables
from app.mcp_server import mcp, user_access_token_var
from app.setup_routes import setup_routes


# ---------------------------------------------------------------------------
# Custom ASGI router for per-user MCP URLs
# ---------------------------------------------------------------------------

class GarminMCPRouter:
    """
    Routes requests from:
      /mcp/{access_token}/sse         → FastMCP SSE endpoint (GET)
      /mcp/{access_token}/messages/   → FastMCP messages endpoint (POST)

    Extracts {access_token} from the URL, sets it in user_access_token_var
    (a contextvars.ContextVar), rewrites the path, and delegates to the
    FastMCP Starlette SSE app.

    This design avoids Starlette's Mount path-parameter edge cases and gives
    us full control over the path rewriting and context injection.
    """

    MCP_PREFIX = "/mcp/"
    VALID_SUBPATHS = {"/sse", "/messages/", "/messages"}

    def __init__(self):
        # Get the Starlette ASGI app from FastMCP
        self._mcp_app = mcp.sse_app()

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            # Pass lifespan events through to the MCP app
            await self._mcp_app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        if not path.startswith(self.MCP_PREFIX):
            # Not an MCP path — return 404
            await self._not_found(scope, receive, send)
            return

        # Parse: /mcp/{access_token}/{subpath}
        rest = path[len(self.MCP_PREFIX):]  # "{access_token}/sse" or "{access_token}/messages/"
        slash_idx = rest.find("/")

        if slash_idx == -1:
            # No subpath — invalid
            await self._not_found(scope, receive, send)
            return

        access_token = rest[:slash_idx]
        sub_path = rest[slash_idx:]  # "/sse" or "/messages/"

        if not access_token:
            await self._not_found(scope, receive, send)
            return

        if sub_path not in self.VALID_SUBPATHS:
            await self._not_found(scope, receive, send)
            return

        # Set the ContextVar so MCP tools can read the user's identity
        token_ctx = user_access_token_var.set(access_token)

        # Rewrite path so FastMCP sees /sse or /messages/
        new_scope = dict(scope)
        new_scope["path"] = sub_path
        new_scope["raw_path"] = sub_path.encode()

        try:
            await self._mcp_app(new_scope, receive, send)
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


async def mcp_endpoint(scope, receive, send):
    """ASGI passthrough to GarminMCPRouter (used in Starlette Mount)."""
    await _mcp_router(scope, receive, send)


async def on_startup():
    """Create database tables on startup (idempotent — uses CREATE IF NOT EXISTS)."""
    await create_tables()


app = Starlette(
    on_startup=[on_startup],
    routes=[
        # Web pages + API endpoints
        *setup_routes,

        # MCP connector — everything under /mcp/ goes to GarminMCPRouter
        Mount("/mcp", app=_mcp_router),
    ],
)
