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
# API: Setup — Step 1 (email + password)
# ---------------------------------------------------------------------------

async def api_setup_start(request: Request) -> JSONResponse:
    """
    Begin Garmin authentication.

    Request body: {"email": str, "password": str}

    Responses:
      200 {"mcp_url": str}                       — success, no MFA
      200 {"mfa_required": true, "session_id": str} — MFA required
      400 {"error": str}                          — bad credentials or other error
    """
    from garminconnect import Garmin, GarminConnectAuthenticationError

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    email = (body.get("email") or "").strip()
    password = body.get("password") or ""

    if not email or not password:
        return JSONResponse({"error": "Email and password are required"}, status_code=400)

    # Create Garmin client with return_on_mfa=True so we can handle MFA in-browser
    garmin_client = Garmin(email=email, password=password)

    loop = asyncio.get_event_loop()

    try:
        # Run blocking login in thread executor
        result = await loop.run_in_executor(
            None,
            lambda: _attempt_login(garmin_client),
        )
    except GarminConnectAuthenticationError as e:
        return JSONResponse(
            {"error": f"Invalid Garmin credentials: {e}"},
            status_code=400,
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"Authentication failed: {e}"},
            status_code=400,
        )

    if result.get("mfa_required"):
        session_id = create_mfa_session(
            garmin_client=garmin_client,
            client_state=result["client_state"],
            email=email,
        )
        return JSONResponse({"mfa_required": True, "session_id": session_id})

    # No MFA — tokens are on garmin_client.garth
    token_b64 = garmin_client.garth.dumps()
    display_name = getattr(garmin_client, "display_name", None)
    mcp_url = await _save_user_and_get_url(request, token_b64, display_name, email)
    return JSONResponse({"mcp_url": mcp_url})


def _attempt_login(garmin_client) -> dict:
    """
    Synchronous Garmin login with MFA detection.
    Returns {"mfa_required": False} or {"mfa_required": True, "client_state": ...}

    garminconnect's login() raises an exception or calls an MFA callback.
    We monkey-patch the MFA callback to intercept the flow.
    """
    mfa_result = {"mfa_required": False}

    def mfa_callback(prompt: str) -> str:
        """Called by garth when MFA code is needed."""
        # Store state so we can resume later; raise to abort this login attempt
        mfa_result["mfa_required"] = True
        mfa_result["client_state"] = {"client": garmin_client.garth}
        # Return empty string — garth will fail, but we catch that externally
        return "__MFA_INTERCEPTED__"

    try:
        garmin_client.login(mfa_token_callback=mfa_callback)
    except Exception as e:
        # If MFA was intercepted, the login will fail — that's expected
        if mfa_result["mfa_required"]:
            return mfa_result
        raise  # Re-raise genuine errors

    return mfa_result


# ---------------------------------------------------------------------------
# API: Setup — Step 2 (MFA code)
# ---------------------------------------------------------------------------

async def api_setup_mfa(request: Request) -> JSONResponse:
    """
    Complete Garmin authentication with MFA code.

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

    loop = asyncio.get_event_loop()
    garmin_client = session.garmin_client

    try:
        # Resume login with the MFA code — uses the live garth.Client with SSO cookies
        await loop.run_in_executor(
            None,
            lambda: garmin_client.login(mfa_token_callback=lambda _: mfa_code),
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"MFA authentication failed: {e}. Check the code and try again."},
            status_code=400,
        )

    token_b64 = garmin_client.garth.dumps()
    display_name = getattr(garmin_client, "display_name", None)
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
