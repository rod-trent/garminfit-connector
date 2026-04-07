"""
Web routes for the user-facing setup and disconnect flows.

Setup flow
----------
1. GET  /setup                  — instructions + script download
2. GET  /download/garmin_setup  — download the local auth script
3. POST /api/setup/import-token — register session from the local script
4. POST /api/disconnect         — revoke by email

Other routes
------------
GET  /            → redirect to /setup
GET  /disconnect  → disconnect form
GET  /health      → Railway health check
GET  /debug/mcp   → MCP session diagnostics
"""

import asyncio  # used by api_setup_import_token
import logging
import os
import secrets
import time
from datetime import datetime
from typing import Dict, Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Route

from app.auth_manager import encrypt_token, generate_access_token
from app.database import SessionLocal, User
from app.garmin_api_client import GarminApiClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Temporary MFA session storage for the garmy login flow
#
# {session_id: {"auth_client": AuthClient, "client_state": dict,
#               "email": str, "created_at": float}}
# Entries are pruned after MFA_SESSION_TTL seconds.
# ---------------------------------------------------------------------------
_garmy_mfa_sessions: Dict[str, Dict[str, Any]] = {}
MFA_SESSION_TTL = 300  # 5 minutes


def _prune_mfa_sessions() -> None:
    """Remove stale MFA sessions."""
    now = time.monotonic()
    stale = [k for k, v in _garmy_mfa_sessions.items()
             if now - v["created_at"] > MFA_SESSION_TTL]
    for k in stale:
        _garmy_mfa_sessions.pop(k, None)


# ---------------------------------------------------------------------------
# Jinja2 template environment
# ---------------------------------------------------------------------------

_templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
_jinja_env = Environment(
    loader=FileSystemLoader(_templates_dir),
    autoescape=select_autoescape(["html"]),
)


def _render(template_name: str, **ctx) -> HTMLResponse:
    tmpl = _jinja_env.get_template(template_name)
    return HTMLResponse(tmpl.render(**ctx))


# ---------------------------------------------------------------------------
# Helper: save authenticated user and return their MCP URL
# ---------------------------------------------------------------------------

async def _save_user_and_get_url(
    request: Request,
    token_json: str,
    display_name: str | None,
    email: str,
) -> str:
    access_token = generate_access_token()
    encrypted = encrypt_token(token_json)

    base_url = os.environ.get("APP_BASE_URL", str(request.base_url).rstrip("/"))
    mcp_url = f"{base_url}/garmin/?token={access_token}"

    async with SessionLocal() as db:
        user = User(
            access_token=access_token,
            garth_token_encrypted=encrypted,
            display_name=display_name,
            garmin_email=email.lower().strip(),
            created_at=datetime.utcnow(),
        )
        db.add(user)
        await db.commit()

    return mcp_url


# ---------------------------------------------------------------------------
# HTML page routes
# ---------------------------------------------------------------------------

async def root(request: Request):
    return RedirectResponse(url="/setup")


async def setup_page(request: Request) -> HTMLResponse:
    return _render("setup.html")


async def disconnect_page(request: Request) -> HTMLResponse:
    return _render("disconnect.html")


async def setup_success_page(request: Request) -> HTMLResponse:
    return _render("success.html")


async def download_garmin_setup(request: Request) -> FileResponse:
    """Serve the local auth script as a file download."""
    script_path = os.path.join(
        os.path.dirname(__file__), "..", "scripts", "garmin_setup.py"
    )
    script_path = os.path.abspath(script_path)
    return FileResponse(
        path=script_path,
        filename="garmin_setup.py",
        media_type="text/x-python",
    )


async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "garminfit-connector"})


async def debug_mcp(request: Request) -> JSONResponse:
    """Diagnostic endpoint — confirms the MCP session manager is alive."""
    from app.mcp_server import mcp
    try:
        sm = mcp.session_manager
        return JSONResponse({
            "status": "ok",
            "session_manager": type(sm).__name__,
            "json_response": sm.json_response,
            "stateless": sm.stateless,
            "active_sessions": len(getattr(sm, "_server_instances", {})),
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# POST /api/setup/import-token  — for the garmin_givemydata / local scripts
# ---------------------------------------------------------------------------

async def api_setup_import_token(request: Request) -> JSONResponse:
    """
    Register a Garmin session obtained externally (e.g. garmin_givemydata,
    scripts/playwright_setup.py, or any tool that exports Garmin cookies).

    Request body: {"email": str, "token": str}
      token — JSON: {"cookies": {name: value, ...}, "display_name": str}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    email = (body.get("email") or "").strip()
    token = (body.get("token") or "").strip()

    if not email or not token:
        return JSONResponse({"error": "email and token are required"}, status_code=400)

    loop = asyncio.get_event_loop()

    def _validate(token_str: str):
        client = GarminApiClient.from_token(token_str)
        try:
            data = client._get("/userprofile-service/socialProfile")
            display_name = (
                data.get("displayName") or data.get("userName")
                if isinstance(data, dict)
                else None
            )
            if display_name:
                client.display_name = display_name
        except Exception as exc:
            log.info("import-token live validation skipped (%s); using display_name from token", exc)
        return client.display_name, client.dumps()

    try:
        display_name, updated_token = await loop.run_in_executor(None, _validate, token)
    except Exception as exc:
        return JSONResponse({"error": f"Token import failed: {exc}"}, status_code=400)

    try:
        mcp_url = await _save_user_and_get_url(request, updated_token, display_name, email)
    except Exception as exc:
        return JSONResponse({"error": f"Session valid but failed to save: {exc}"}, status_code=500)

    return JSONResponse({"mcp_url": mcp_url})


# ---------------------------------------------------------------------------
# POST /api/disconnect
# ---------------------------------------------------------------------------

async def api_disconnect(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    email = (body.get("email") or "").strip().lower()
    if not email:
        return JSONResponse({"error": "Email address is required"}, status_code=400)

    now = datetime.utcnow()
    revoked_count = 0

    async with SessionLocal() as db:
        result = await db.execute(
            select(User).where(
                User.garmin_email == email,
                User.revoked == False,  # noqa: E712
            )
        )
        for user in result.scalars().all():
            user.revoked = True
            user.revoked_at = now
            revoked_count += 1
        await db.commit()

    if revoked_count == 0:
        return JSONResponse(
            {"error": f"No active connections found for {email}."},
            status_code=404,
        )

    return JSONResponse({
        "revoked": revoked_count,
        "message": (
            f"Successfully disconnected {revoked_count} Garmin connection(s) for {email}. "
            "Your MCP URL will no longer work. Visit /setup to reconnect."
        ),
    })


# ---------------------------------------------------------------------------
# POST /api/setup/login  — server-side garmy auth (email + password)
# ---------------------------------------------------------------------------

async def api_setup_login(request: Request) -> JSONResponse:
    """
    Authenticate directly on the server using garmy's OAuth flow.
    No browser or local script required.

    Request body: {"email": str, "password": str}

    Responses:
      {"mcp_url": "..."}                          — login succeeded
      {"mfa_required": true, "session_id": "..."}  — MFA code needed
      {"error": "..."}                             — failure
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    email = (body.get("email") or "").strip()
    password = (body.get("password") or "").strip()

    if not email or not password:
        return JSONResponse(
            {"error": "email and password are required"}, status_code=400
        )

    _prune_mfa_sessions()

    loop = asyncio.get_event_loop()

    def _do_login():
        from garmy import AuthClient, APIClient
        auth_client = AuthClient()
        result = auth_client.login(email, password, return_on_mfa=True)
        return auth_client, result

    try:
        auth_client, result = await loop.run_in_executor(None, _do_login)
    except Exception as exc:
        log.warning("garmy login failed for %s: %s", email, exc)
        return JSONResponse({"error": f"Login failed: {exc}"}, status_code=401)

    # --- MFA required ---
    if isinstance(result, tuple) and result[0] == "needs_mfa":
        _, client_state = result
        session_id = secrets.token_urlsafe(16)
        _garmy_mfa_sessions[session_id] = {
            "auth_client": auth_client,
            "client_state": client_state,
            "email": email,
            "created_at": time.monotonic(),
        }
        return JSONResponse({"mfa_required": True, "session_id": session_id})

    # --- Login succeeded without MFA ---
    return await _garmy_finish_auth(request, auth_client, email)


# ---------------------------------------------------------------------------
# POST /api/setup/mfa  — submit MFA code for a pending garmy login
# ---------------------------------------------------------------------------

async def api_setup_mfa(request: Request) -> JSONResponse:
    """
    Complete a garmy login that required an MFA code.

    Request body: {"session_id": str, "mfa_code": str}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    session_id = (body.get("session_id") or "").strip()
    mfa_code = (body.get("mfa_code") or "").strip()

    if not session_id or not mfa_code:
        return JSONResponse(
            {"error": "session_id and mfa_code are required"}, status_code=400
        )

    session = _garmy_mfa_sessions.pop(session_id, None)
    if not session:
        return JSONResponse(
            {"error": "MFA session not found or expired. Please log in again."},
            status_code=400,
        )

    auth_client = session["auth_client"]
    client_state = session["client_state"]
    email = session["email"]

    loop = asyncio.get_event_loop()

    def _do_mfa():
        auth_client.resume_login(mfa_code, client_state)
        return auth_client

    try:
        auth_client = await loop.run_in_executor(None, _do_mfa)
    except Exception as exc:
        log.warning("garmy MFA failed for %s: %s", email, exc)
        return JSONResponse({"error": f"MFA failed: {exc}"}, status_code=401)

    return await _garmy_finish_auth(request, auth_client, email)


# ---------------------------------------------------------------------------
# Shared helper: build and store the garmy session
# ---------------------------------------------------------------------------

async def _garmy_finish_auth(
    request: Request,
    auth_client,
    email: str,
) -> JSONResponse:
    """
    After successful garmy auth (with or without MFA): load display_name,
    serialise tokens, persist to DB, return the MCP URL.
    """
    from garmy import APIClient
    from app.garmy_client import GarmyApiClient

    loop = asyncio.get_event_loop()

    def _build_client():
        api_client = APIClient(auth_client=auth_client)
        gc = GarmyApiClient(auth_client, api_client)
        try:
            gc.get_full_name()  # populates gc.display_name
        except Exception:
            pass
        return gc

    try:
        gc = await loop.run_in_executor(None, _build_client)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to initialise API client: {exc}"}, status_code=500
        )

    token_json = gc.dumps()

    try:
        mcp_url = await _save_user_and_get_url(
            request, token_json, gc.display_name or None, email
        )
    except Exception as exc:
        return JSONResponse(
            {"error": f"Auth succeeded but failed to save session: {exc}"},
            status_code=500,
        )

    return JSONResponse({"mcp_url": mcp_url})


# ---------------------------------------------------------------------------
# Route list (imported by main.py)
# ---------------------------------------------------------------------------

setup_routes = [
    Route("/", root, methods=["GET"]),
    Route("/setup", setup_page, methods=["GET"]),
    Route("/setup/success", setup_success_page, methods=["GET"]),
    Route("/disconnect", disconnect_page, methods=["GET"]),
    Route("/health", health_check, methods=["GET"]),
    Route("/debug/mcp", debug_mcp, methods=["GET"]),
    Route("/download/garmin_setup.py", download_garmin_setup, methods=["GET"]),
    Route("/api/setup/import-token", api_setup_import_token, methods=["POST"]),
    Route("/api/setup/login", api_setup_login, methods=["POST"]),
    Route("/api/setup/mfa", api_setup_mfa, methods=["POST"]),
    Route("/api/disconnect", api_disconnect, methods=["POST"]),
]
