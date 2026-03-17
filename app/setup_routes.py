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

MFA design (v2 — garth return_on_mfa)
--------------------------------------
garth 0.6+ provides sso.login(return_on_mfa=True) which returns immediately
with a client_state dict when MFA is required, and sso.resume_login() which
completes the flow given that state + the user's code.

This replaces the old MFABridge pattern that kept a thread blocked at
prompt_mfa() while the user typed.  The old approach raced against the CAS
service ticket TTL: the SSO session was held open across two HTTP round-trips
(server→browser and browser→server), and if the ticket expired before the
OAuth preauthorized exchange ran, Garmin returned 401.

New flow:
  1. api_setup_start calls sso.login(..., return_on_mfa=True) in a thread.
     If MFA is needed, it returns ("needs_mfa", client_state) immediately —
     no thread is left blocking.  We store client_state and return
     {mfa_required: true, session_id} to the browser.
  2. api_setup_mfa retrieves the stored client_state and calls
     sso.resume_login(client_state, mfa_code) in a thread.  This submits
     the code to Garmin's SSO, gets the ticket, and exchanges it for OAuth
     tokens — all in one atomic blocking call with no added latency.

Each login uses its own isolated garth.http.Client instance, so concurrent
users never share global garth state.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from garth import sso as garth_sso
from garth.http import Client as GarthClient
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
# Thread pool for blocking garth SSO calls
# ---------------------------------------------------------------------------

# sso.login() and sso.resume_login() are synchronous (requests-based).
# We run them in this pool so they don't block the asyncio event loop.
# With return_on_mfa, no thread is ever held waiting for user input, so
# a small pool is sufficient.
_login_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="garmin-login")

# ---------------------------------------------------------------------------
# Rate limiting — pace outgoing SSO requests to Garmin.
# All Railway dynos share one outgoing IP; Garmin rate-limits by IP.
# We serialise the *initial* SSO phase with a lock + minimum interval.
# MFA completion (resume_login) does NOT need the lock — it's a follow-up
# call on an existing session, not a fresh credential submission.
# ---------------------------------------------------------------------------

_setup_lock = asyncio.Lock()
_MIN_LOGIN_INTERVAL_SECS: float = 4.0
_last_login_time: float = 0.0


def _is_rate_limited(exc: Exception) -> bool:
    """Return True if the exception is a Garmin SSO 429 / rate-limit error."""
    msg = str(exc)
    return "429" in msg or "Too Many Requests" in msg or "rate" in msg.lower()


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
    Begin Garmin authentication using an isolated garth.http.Client.

    Each call creates its own GarthClient so that concurrent logins never
    share global garth state.  An MFABridge is injected as the prompt_mfa
    callback; if Garmin requires MFA the login thread blocks inside
    bridge.prompt_mfa() until api_setup_mfa() provides the code.

    Request body: {"email": str, "password": str}

    Responses:
      200 {"mcp_url": str}                       — success, no MFA
      200 {"mfa_required": true, "session_id": str} — MFA required
      400 {"error": str}                          — bad credentials or other error
      429 {"error": str, "rate_limited": true, "retry_after": 60}
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

    # --- Rate-limit phase: hold lock only for the initial SSO request ------
    # Lock is released as soon as sso.login() returns (either with tokens or
    # "needs_mfa").  MFA completion runs outside the lock via resume_login().
    async with _setup_lock:
        global _last_login_time

        elapsed = time.monotonic() - _last_login_time
        wait_secs = _MIN_LOGIN_INTERVAL_SECS - elapsed
        if wait_secs > 0:
            await asyncio.sleep(wait_secs)

        isolated_client = GarthClient()

        # sso.login(return_on_mfa=True) returns immediately in both cases:
        #   - No MFA:  (OAuth1Token, OAuth2Token)
        #   - MFA req: ("needs_mfa", {"signin_params": ..., "client": ...})
        # No thread is left blocking — the lock can be released right away.
        try:
            result = await loop.run_in_executor(
                _login_executor,
                lambda: garth_sso.login(
                    email, password,
                    return_on_mfa=True,
                    client=isolated_client,
                ),
            )
        except Exception as e:
            print(f"[setup] sso.login error: {type(e).__name__}: {e}")
            _last_login_time = time.monotonic()
            if _is_rate_limited(e):
                return JSONResponse(
                    {
                        "error": (
                            "Garmin is temporarily rate-limiting new connections from this server. "
                            "Please wait 60 seconds and try again."
                        ),
                        "rate_limited": True,
                        "retry_after": 60,
                    },
                    status_code=429,
                )
            return JSONResponse(
                {"error": f"Authentication failed: {e}"},
                status_code=400,
            )

        _last_login_time = time.monotonic()
        # Lock released here.

    # --- Interpret result ---------------------------------------------------

    if result[0] == "needs_mfa":
        # Store the garth client_state so resume_login() can complete it.
        client_state = result[1]
        session_id = create_mfa_session(
            garmin_email=email,
            client_state=client_state,
            isolated_client=isolated_client,
        )
        return JSONResponse({"mfa_required": True, "session_id": session_id})

    # No MFA — result is (OAuth1Token, OAuth2Token)
    oauth1_token, oauth2_token = result
    isolated_client.configure(
        oauth1_token=oauth1_token,
        oauth2_token=oauth2_token,
        domain=oauth1_token.domain,
    )

    try:
        token_b64 = isolated_client.dumps()
    except Exception as e:
        print(f"[setup] dumps() error: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Garmin auth succeeded but failed to serialize token: {e}"},
            status_code=500,
        )

    display_name = await _get_display_name(token_b64)

    try:
        mcp_url = await _save_user_and_get_url(request, token_b64, display_name, email)
    except Exception as e:
        print(f"[setup] save_user error: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Garmin auth succeeded but failed to save your account: {e}"},
            status_code=500,
        )

    return JSONResponse({"mcp_url": mcp_url})


# ---------------------------------------------------------------------------
# API: Setup — Step 2 (MFA code)
# ---------------------------------------------------------------------------

async def api_setup_mfa(request: Request) -> JSONResponse:
    """
    Complete Garmin authentication with MFA code.

    Retrieves the stored garth client_state and calls sso.resume_login()
    which submits the MFA code to Garmin's SSO, extracts the CAS ticket,
    and exchanges it for OAuth tokens — all in one atomic call with no
    additional latency between the MFA submission and the OAuth exchange.

    Request body: {"session_id": str, "mfa_code": str}

    Responses:
      200 {"mcp_url": str}                        — success
      400 {"error": str, "restart_required": bool} — expired/wrong code, etc.
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
            {
                "error": "Session expired or not found. Please start the setup process again.",
                "restart_required": True,
            },
            status_code=400,
        )

    client_state = session.client_state
    isolated_client = session.isolated_client
    loop = asyncio.get_event_loop()

    # resume_login() submits the MFA code to SSO, gets the CAS ticket, and
    # exchanges it for OAuth tokens in one uninterrupted synchronous call.
    # Because there is no gap between the MFA submission and the OAuth
    # exchange the CAS ticket cannot expire in transit.
    try:
        oauth1_token, oauth2_token = await asyncio.wait_for(
            loop.run_in_executor(
                _login_executor,
                lambda: garth_sso.resume_login(client_state, mfa_code),
            ),
            timeout=60,
        )
    except asyncio.TimeoutError:
        print(f"[MFA] resume_login timed out for session {session_id}")
        return JSONResponse(
            {
                "error": "MFA authentication timed out. Please start the setup process again.",
                "restart_required": True,
            },
            status_code=400,
        )
    except Exception as e:
        print(f"[MFA] resume_login error: {type(e).__name__}: {e}")
        error_str = str(e)

        # 401 typically means a wrong or already-used TOTP code.
        if "401" in error_str:
            return JSONResponse(
                {
                    "error": (
                        "Invalid or expired MFA code. "
                        "Please enter the latest code from your authenticator app and try again."
                    ),
                    "restart_required": False,
                },
                status_code=400,
            )

        # All other failures — session state is unknown, restart required.
        return JSONResponse(
            {
                "error": f"MFA authentication failed: {e}. Please start the setup process again.",
                "restart_required": True,
            },
            status_code=400,
        )

    # Attach tokens to isolated_client so we can call dumps()
    isolated_client.configure(
        oauth1_token=oauth1_token,
        oauth2_token=oauth2_token,
        domain=oauth1_token.domain,
    )

    try:
        token_b64 = isolated_client.dumps()
        print(f"[MFA] token serialized OK, length={len(token_b64)}")
    except Exception as e:
        print(f"[MFA] dumps() error: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Garmin auth succeeded but failed to serialize token: {e}"},
            status_code=500,
        )

    display_name = await _get_display_name(token_b64)
    try:
        mcp_url = await _save_user_and_get_url(
            request, token_b64, display_name, session.garmin_email
        )
    except Exception as e:
        print(f"[MFA] save_user error: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Garmin auth succeeded but failed to save: {e}"},
            status_code=500,
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


async def debug_mcp(request: Request) -> JSONResponse:
    """
    Diagnostic endpoint — confirms the MCP session manager is alive.
    Visit /debug/mcp to verify the server is healthy before connecting Claude.
    """
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


setup_routes = [
    Route("/", root, methods=["GET"]),
    Route("/setup", setup_page, methods=["GET"]),
    Route("/setup/success", setup_success_page, methods=["GET"]),
    Route("/disconnect", disconnect_page, methods=["GET"]),
    Route("/health", health_check, methods=["GET"]),
    Route("/debug/mcp", debug_mcp, methods=["GET"]),
    Route("/api/setup/start", api_setup_start, methods=["POST"]),
    Route("/api/setup/mfa", api_setup_mfa, methods=["POST"]),
    Route("/api/disconnect", api_disconnect, methods=["POST"]),
]
