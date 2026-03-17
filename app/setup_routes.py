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

MFA design
----------
garth's login() is fully synchronous.  When MFA is required it calls a
`prompt_mfa` callable to collect the code.  We inject an MFABridge as that
callable; it blocks the login thread until the user submits the code in a
second HTTP request (api_setup_mfa).

Each login uses its own isolated garth.http.Client instance, so concurrent
users never share global garth state and there is no risk of one user's SSO
session corrupting another's.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from garth.http import Client as GarthClient
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.auth_manager import (
    MFABridge,
    create_mfa_session,
    encrypt_token,
    generate_access_token,
    pop_mfa_session,
)
from app.database import SessionLocal, User

# ---------------------------------------------------------------------------
# Thread pool for blocking garth login calls
# ---------------------------------------------------------------------------

# Each pending MFA session holds one thread (blocked at prompt_mfa).
# 20 workers lets us handle 20 simultaneous in-flight logins.
_login_executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="garmin-login")

# ---------------------------------------------------------------------------
# Rate limiting — pace outgoing SSO requests to Garmin.
# All Railway users share one outgoing IP; Garmin rate-limits by IP.
# We serialise the *initial* SSO phase with a lock + minimum interval.
# Once a thread is blocked at the MFA prompt it holds no lock, so other
# users can continue logging in while the first user types their code.
# ---------------------------------------------------------------------------

_setup_lock = asyncio.Lock()
_MIN_LOGIN_INTERVAL_SECS: float = 4.0
_last_login_time: float = 0.0


def _is_rate_limited(exc: Exception) -> bool:
    """Return True if the exception is a Garmin SSO 429 / rate-limit error."""
    msg = str(exc)
    return "429" in msg or "Too Many Requests" in msg or "rate" in msg.lower()


async def _wait_login_or_mfa(
    login_future,
    bridge: MFABridge,
    loop: asyncio.AbstractEventLoop,
    timeout: float = 60.0,
) -> str:
    """
    Poll until the login thread either completes or triggers the MFA prompt.

    Returns:
      "mfa"     — bridge.mfa_triggered is set (thread is blocked at prompt_mfa)
      "done"    — login_future has completed (success or exception)
      "timeout" — neither happened within `timeout` seconds
    """
    wrapped = asyncio.wrap_future(login_future, loop=loop)
    deadline = loop.time() + timeout

    while loop.time() < deadline:
        if bridge.mfa_triggered.is_set():
            return "mfa"
        if wrapped.done():
            return "done"
        try:
            await asyncio.wait_for(asyncio.shield(wrapped), timeout=0.15)
            return "done"
        except asyncio.TimeoutError:
            continue

    return "timeout"


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
    async with _setup_lock:
        global _last_login_time

        # Enforce minimum interval between sequential logins
        elapsed = time.monotonic() - _last_login_time
        wait_secs = _MIN_LOGIN_INTERVAL_SECS - elapsed
        if wait_secs > 0:
            await asyncio.sleep(wait_secs)

        isolated_client = GarthClient()
        bridge = MFABridge()

        # Submit login to thread pool — the thread may block inside
        # bridge.prompt_mfa() if MFA is required.
        login_future = _login_executor.submit(
            lambda: isolated_client.login(email, password, prompt_mfa=bridge.prompt_mfa)
        )

        # Wait until the thread either reaches the MFA prompt or finishes.
        # While we wait we still hold _setup_lock so only one SSO handshake
        # is in flight at a time.
        outcome = await _wait_login_or_mfa(login_future, bridge, loop, timeout=60.0)

        _last_login_time = time.monotonic()
        # Lock released here — if outcome=="mfa" the login thread is
        # blocked at prompt_mfa() and is no longer making SSO requests.

    # --- Interpret outcome --------------------------------------------------

    if outcome == "mfa":
        session_id = create_mfa_session(
            garmin_email=email,
            mfa_bridge=bridge,
            login_future=login_future,
            isolated_client=isolated_client,
        )
        return JSONResponse({"mfa_required": True, "session_id": session_id})

    # outcome == "done" (or "timeout" — treat as done, let result() raise)
    try:
        login_future.result(timeout=5)
    except Exception as e:
        print(f"[setup] login error: {type(e).__name__}: {e}")
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

    # Success — isolated_client now has oauth1_token + oauth2_token set
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

    Provides the code to the waiting MFABridge, which unblocks the login
    thread.  We then await the thread's future to get the final result and
    extract the token from the isolated GarthClient.

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

    bridge: MFABridge = session.mfa_bridge
    login_future = session.login_future
    isolated_client = session.isolated_client
    loop = asyncio.get_event_loop()

    # Provide the MFA code — this unblocks the login thread
    bridge.submit_code(mfa_code)

    # Await the login thread to complete (up to 60 s)
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, lambda: login_future.result(timeout=60)),
            timeout=65,
        )
    except asyncio.TimeoutError:
        print(f"[MFA] login_future timed out for session {session_id}")
        return JSONResponse(
            {
                "error": "MFA authentication timed out. Please start the setup process again.",
                "restart_required": True,
            },
            status_code=400,
        )
    except Exception as e:
        print(f"[MFA] login error: {type(e).__name__}: {e}")
        error_str = str(e)

        # 401 at Garmin's OAuth preauthorized endpoint means the SSO/CAS ticket
        # expired or Garmin rejected the token exchange (e.g. IP throttling).
        # This is NOT a wrong-code error — the MFA code itself was likely fine,
        # but the underlying login session can no longer be recovered.
        # The only fix is to restart the login flow from scratch.
        if "preauthorized" in error_str and "401" in error_str:
            return JSONResponse(
                {
                    "error": (
                        "Garmin's login session expired before the OAuth token exchange "
                        "could complete. This usually happens when the MFA code takes too "
                        "long to submit or Garmin's servers are busy. "
                        "Please start the setup process again with a fresh code."
                    ),
                    "restart_required": True,
                },
                status_code=400,
            )

        # A 401 at the SSO/CAS layer (not preauthorized) typically means a wrong
        # or already-used MFA code. The user can retry with a new code.
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

        # All other failures (network errors, unexpected API responses, etc.)
        return JSONResponse(
            {
                "error": f"MFA authentication failed: {e}. Please start the setup process again.",
                "restart_required": True,
            },
            status_code=400,
        )

    # isolated_client now has oauth1_token + oauth2_token set by garth
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
