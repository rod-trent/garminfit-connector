"""
Web routes for the user-facing setup and disconnect flows.

Endpoints:
  GET  /            → redirect to /setup
  GET  /setup       → setup form (enter Garmin credentials)
  GET  /disconnect  → disconnect form (enter email to revoke)
  GET  /health      → health check (used by Railway)

  POST /api/setup/start      → begin Garmin auth (email + password)
  POST /api/setup/mfa        → complete MFA (session_id + mfa_code)
  POST /api/disconnect       → revoke user's access token

All API responses are JSON. HTML pages use Jinja2 templates.
"""

import asyncio
import os
from datetime import datetime

import garth
from garth.sso import resume_login as garth_resume_login
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.auth_manager import (
    create_mfa_session,
    encrypt_token,
    generate_access_token,
    pop_mfa_session,
)
from app.database import SessionLocal, User

# Serialize concurrent setup requests: garth uses a module-level global client,
# so two simultaneous logins would overwrite each other's token state.
# Setup is rare (one-off per user) so a simple lock is fine.
_setup_lock = asyncio.Lock()

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
    token_b64: str,
    display_name: str | None,
    email: str,
) -> str:
    access_token = generate_access_token()
    encrypted = encrypt_token(token_b64)

    base_url = os.environ.get("APP_BASE_URL", str(request.base_url).rstrip("/"))
    mcp_url = f"{base_url}/mcp/{access_token}/sse"

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


async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "garminfit-connector"})


# ---------------------------------------------------------------------------
# Helper: get Garmin display name from a token (optional — never crashes setup)
# ---------------------------------------------------------------------------

async def _get_display_name(token_b64: str) -> str | None:
    """
    Create a temporary Garmin client from an existing token and fetch the
    display name. Returns None on any failure — this is purely cosmetic.
    """
    try:
        from garminconnect import Garmin
        loop = asyncio.get_event_loop()
        client = Garmin()
        await loop.run_in_executor(None, lambda: client.login(tokenstore=token_b64))
        return getattr(client, "display_name", None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# API: Setup — Step 1 (email + password)
# ---------------------------------------------------------------------------

async def api_setup_start(request: Request) -> JSONResponse:
    """
    Begin Garmin authentication using garth's low-level API directly.

    garth.login(email, password, return_on_mfa=True) returns:
      - A falsy value / None  →  success, no MFA needed
      - A tuple (oauth1_token, client_state)  →  MFA required

    Request body: {"email": str, "password": str}

    Responses:
      200 {"mcp_url": str}                       — success, no MFA
      200 {"mfa_required": true, "session_id": str} — MFA required
      400 {"error": str}                          — bad credentials or other error
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    email = (body.get("email") or "").strip()
    password = body.get("password") or ""

    if not email or not password:
        return JSONResponse({"error": "Email and password are required"}, status_code=400)

    loop = asyncio.get_event_loop()

    # Serialize login requests — garth uses a module-level global client
    async with _setup_lock:
        try:
            result = await loop.run_in_executor(
                None,
                lambda: garth.login(email, password, return_on_mfa=True),
            )
        except Exception as e:
            return JSONResponse(
                {"error": f"Authentication failed: {e}"},
                status_code=400,
            )

        if isinstance(result, tuple) and len(result) == 2:
            # MFA required — result is (oauth1_token, client_state)
            _oauth1_token, client_state = result
            session_id = create_mfa_session(
                garmin_client=None,
                client_state=client_state,
                email=email,
            )
            return JSONResponse({"mfa_required": True, "session_id": session_id})

        # No MFA — garth.client now holds the tokens
        token_b64 = garth.client.dumps()

    # DB save and optional display_name lookup — outside the lock
    display_name = await _get_display_name(token_b64)
    mcp_url = await _save_user_and_get_url(request, token_b64, display_name, email)
    return JSONResponse({"mcp_url": mcp_url})


# ---------------------------------------------------------------------------
# API: Setup — Step 2 (MFA code)
# ---------------------------------------------------------------------------

async def api_setup_mfa(request: Request) -> JSONResponse:
    """
    Complete Garmin authentication with MFA code.

    Uses garth.sso.resume_login(client_state, mfa_code) to complete the flow
    started in api_setup_start, then extracts the final token via garth.client.dumps().

    Request body: {"session_id": str, "mfa_code": str}

    Responses:
      200 {"mcp_url": str}  — success
      400 {"error": str}    — expired session, wrong code, etc.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    session_id = (body.get("session_id") or "").strip()
    mfa_code = (body.get("mfa_code") or "").strip().replace(" ", "")

    if not session_id or not mfa_code:
        return JSONResponse({"error": "session_id and mfa_code are required"}, status_code=400)

    session = pop_mfa_session(session_id)
    if not session:
        return JSONResponse(
            {"error": "Session expired or not found. Please start the setup process again."},
            status_code=400,
        )

    client_state = session.client_state
    loop = asyncio.get_event_loop()

    async with _setup_lock:
        try:
            # garth.sso.resume_login takes the client_state from step 1 and the MFA code,
            # returns (oauth1_token, oauth2_token) on success
            oauth1_token, oauth2_token = await loop.run_in_executor(
                None,
                lambda: garth_resume_login(client_state, mfa_code),
            )
        except Exception as e:
            return JSONResponse(
                {"error": f"MFA authentication failed: {e}. Check the code and try again."},
                status_code=400,
            )

        # Inject the tokens into garth's global client and serialize them
        garth.client.oauth1_token = oauth1_token
        garth.client.oauth2_token = oauth2_token
        token_b64 = garth.client.dumps()

    # DB save and optional display_name lookup — outside the lock
    display_name = await _get_display_name(token_b64)
    mcp_url = await _save_user_and_get_url(
        request, token_b64, display_name, session.garmin_email
    )
    return JSONResponse({"mcp_url": mcp_url})


# ---------------------------------------------------------------------------
# API: Disconnect (revoke access token)
# ---------------------------------------------------------------------------

async def api_disconnect(request: Request) -> JSONResponse:
    """
    Revoke all access tokens for a given Garmin email address.

    Request body: {"email": str}

    Responses:
      200 {"revoked": int}  — number of tokens revoked
      400 {"error": str}
    """
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
        users = result.scalars().all()
        for user in users:
            user.revoked = True
            user.revoked_at = now
            revoked_count += 1
        await db.commit()

    if revoked_count == 0:
        return JSONResponse(
            {"error": f"No active connections found for {email}."},
            status_code=404,
        )

    return JSONResponse(
        {
            "revoked": revoked_count,
            "message": (
                f"Successfully disconnected {revoked_count} Garmin connection(s) for {email}. "
                "Your MCP URL will no longer work. Visit /setup to reconnect."
            ),
        }
    )


# ---------------------------------------------------------------------------
# Route list (imported by main.py)
# ---------------------------------------------------------------------------

from starlette.routing import Route


async def setup_success_page(request: Request) -> HTMLResponse:
    """Serve the success page (shows MCP URL from JS query param)."""
    return _render("success.html")


setup_routes = [
    Route("/", root, methods=["GET"]),
    Route("/setup", setup_page, methods=["GET"]),
    Route("/setup/success", setup_success_page, methods=["GET"]),
    Route("/disconnect", disconnect_page, methods=["GET"]),
    Route("/health", health_check, methods=["GET"]),
    Route("/api/setup/start", api_setup_start, methods=["POST"]),
    Route("/api/setup/mfa", api_setup_mfa, methods=["POST"]),
    Route("/api/disconnect", api_disconnect, methods=["POST"]),
]
