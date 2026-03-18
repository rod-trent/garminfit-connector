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

MFA design (v4 — native garth 0.7.9 resume_login)
---------------------------------------------------
garth 0.7.9 rewrites the SSO flow to use Garmin's mobile JSON API
(/mobile/api/login, /mobile/api/mfa/verifyCode) and carries Cloudflare
load-balancer cookies into the OAuth1 preauthorized exchange via a
/portal/sso/embed step.  This resolves both the legacy-account 401 (fixed
by switching to Android consumer-key constants) and the MFA cookie problem
that previously required our manual _resume_login_with_cookies workaround.

The flow is:
  sso.login(email, password, return_on_mfa=True, client=isolated_client)
    → ("needs_mfa", client_state)   # MFA path
    → (OAuth1Token, OAuth2Token)    # no-MFA path

  sso.resume_login(client_state, mfa_code)
    → (OAuth1Token, OAuth2Token)    # submits code, exchanges token atomically

Each login uses its own isolated garth.http.Client instance, so concurrent
users never share global garth state.
"""

import asyncio
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from garth import sso as garth_sso
from garth.auth_tokens import OAuth1Token as _OAuth1Token
from garth.exc import GarthException, GarthHTTPError
from garth.http import Client as GarthClient
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

from app.auth_manager import (
    create_mfa_session,
    encrypt_token,
    generate_access_token,
    get_mfa_session,
    remove_mfa_session,
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
# 10-second gap between sequential SSO logins from this server's shared IP.
# Garmin's OAuth preauthorized endpoint has a per-IP rate limit that can
# impose windows of 15–30 minutes; spacing requests out reduces how quickly
# we burn through the budget.
_MIN_LOGIN_INTERVAL_SECS: float = 10.0
_last_login_time: float = 0.0

# ---------------------------------------------------------------------------
# Server-wide Garmin IP block state
# ---------------------------------------------------------------------------
# When Garmin returns a 429 or an IP-level 401 at the SSO signin endpoint,
# we record a deadline here.  Every subsequent /api/setup/start request checks
# this BEFORE acquiring the lock or touching Garmin at all.
#
# Without this gate, each new user who visits the setup page fires another
# login attempt, which burns more of the per-IP rate-limit budget and makes
# the ban window progressively longer — turning a 30-minute ban into hours.
_garmin_blocked_until: float = 0.0   # time.monotonic() deadline
_garmin_blocked_reason: str = ""


def _set_garmin_block(seconds: float, reason: str = "") -> None:
    """
    Record that Garmin has rate-limited or blocked this server's IP.

    Never shortens an existing block — if we already know we're blocked
    for 30 minutes, a new 5-minute 429 doesn't reduce the deadline.
    """
    global _garmin_blocked_until, _garmin_blocked_reason
    deadline = time.monotonic() + seconds
    if deadline > _garmin_blocked_until:
        _garmin_blocked_until = deadline
        _garmin_blocked_reason = reason
    print(
        f"[rate-limit] Garmin IP block set: {reason!r} — "
        f"{int(seconds)}s cooldown"
    )


def _garmin_block_remaining() -> float:
    """Return seconds until the current block expires (0.0 if not blocked)."""
    return max(0.0, _garmin_blocked_until - time.monotonic())


def _is_rate_limited(exc: Exception) -> bool:
    """Return True if the exception is a Garmin SSO 429 / rate-limit error."""
    msg = str(exc)
    return "429" in msg or "Too Many Requests" in msg or "rate" in msg.lower()


def _is_ip_blocked(exc: Exception) -> bool:
    """
    Return True if Garmin has blocked our IP at the SSO signin level.

    A 401 at sso.garmin.com/sso/signin is NOT a wrong-password error
    (garth surfaces credential failures differently); it means Garmin has
    rejected the IP entirely — a more severe escalation than a 429.
    """
    msg = str(exc)
    return "401" in msg and "sso/signin" in msg



def _extract_retry_after(exc: Exception, default: int = 1800) -> int:
    """
    Extract the Retry-After seconds from a GarthHTTPError 429 response.

    Garmin sometimes returns a Retry-After header telling clients exactly
    how long to wait.  Falls back to `default` (30 min) when absent.
    """
    try:
        if isinstance(exc, GarthHTTPError) and exc.error is not None:
            resp = getattr(exc.error, "response", None)
            if resp is not None:
                ra = resp.headers.get("Retry-After", "")
                print(f"[rate-limit] Retry-After header: {ra!r}")
                if ra.isdigit():
                    return int(ra)
    except Exception:
        pass
    return default


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
    remaining = _garmin_block_remaining()
    payload: dict = {"status": "ok", "service": "garminfit-connector"}
    if remaining > 0:
        payload["garmin_ip_block"] = {
            "blocked": True,
            "reason": _garmin_blocked_reason,
            "retry_after_secs": int(remaining),
            "retry_after_mins": math.ceil(remaining / 60),
        }
    return JSONResponse(payload)


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

    # --- Global IP-block gate ----------------------------------------------
    # Check BEFORE acquiring the lock or touching Garmin at all.
    # Each failed attempt deepens the ban; this gate ensures we never
    # retry while we already know the IP is blocked.
    remaining = _garmin_block_remaining()
    if remaining > 0:
        mins = math.ceil(remaining / 60)
        print(f"[setup] Rejecting login attempt — IP blocked for ~{mins}m ({_garmin_blocked_reason})")
        return JSONResponse(
            {
                "error": (
                    f"Garmin's servers are temporarily rate-limiting this server's IP "
                    f"({_garmin_blocked_reason}). "
                    f"Please try again in approximately {mins} minute{'s' if mins != 1 else ''}."
                ),
                "rate_limited": True,
                "retry_after": int(remaining),
            },
            status_code=429,
        )

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

        # Retry only on transient server errors — NOT on 429.
        # Retrying a 429 burns more of Garmin's per-IP rate-limit budget and
        # makes the ban window longer.  When we get 429 we fail fast, surface
        # the Retry-After time to the user, and let them wait it out.
        isolated_client.configure(
            status_forcelist=(408, 500, 502, 503, 504),
            retries=3,
            backoff_factor=1.0,
        )

        # Residential proxy support.
        # Garmin's OAuth endpoints (connectapi.garmin.com/oauth-service/oauth/
        # preauthorized) are protected by Cloudflare.  Requests from data-centre
        # IPs (Railway, AWS, GCP, etc.) are rate-limited or blocked outright.
        # Set SSO_PROXY_URL to a residential proxy to route all SSO + OAuth
        # traffic through a clean IP.  The proxy is only used for the setup
        # flow; ongoing MCP data requests are unaffected.
        # Format: http://user:pass@host:port  or  socks5://user:pass@host:port
        _sso_proxy = os.environ.get("SSO_PROXY_URL", "").strip()
        if _sso_proxy:
            isolated_client.configure(proxies={"https": _sso_proxy, "http": _sso_proxy})
            print(f"[setup] SSO proxy active: {_sso_proxy.split('@')[-1]}")  # log host only
        else:
            print("[setup] No SSO_PROXY_URL set — using direct connection (may hit Cloudflare blocks)")

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
                retry_after = _extract_retry_after(e)
                _set_garmin_block(retry_after, "429 at OAuth preauthorized")
                wait_mins = max(1, (retry_after + 59) // 60)
                return JSONResponse(
                    {
                        "error": (
                            f"Garmin is rate-limiting new connections from this server's IP. "
                            f"No further attempts will be made for ~{wait_mins} minute(s) "
                            f"to avoid extending the ban. Please try again later."
                        ),
                        "rate_limited": True,
                        "retry_after": retry_after,
                    },
                    status_code=429,
                )

            # 401 at the SSO signin URL means Garmin has escalated from
            # rate-limiting the OAuth endpoint to blocking the IP at the
            # SSO layer entirely — a more severe, longer ban.
            if _is_ip_blocked(e):
                _set_garmin_block(1800, "401 at SSO signin — IP-level block")
                return JSONResponse(
                    {
                        "error": (
                            "Garmin has temporarily blocked this server's IP from signing in. "
                            "This is caused by too many recent failed attempts. "
                            "No further attempts will be made for ~30 minutes. "
                            "Please try again later."
                        ),
                        "rate_limited": True,
                        "retry_after": 1800,
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

    session = get_mfa_session(session_id)
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

    # resume_login() submits the MFA code to Garmin's mobile API, gets the
    # CAS ticket, and exchanges it for OAuth tokens in one call.  garth 0.7.9
    # carries cookies into the OAuth1 exchange natively, so no workaround needed.
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
        remove_mfa_session(session_id)
        return JSONResponse(
            {
                "error": "MFA authentication timed out. Please start the setup process again.",
                "restart_required": True,
            },
            status_code=400,
        )
    except Exception as e:
        print(f"[MFA] resume_login error: {type(e).__name__}: {e}")

        # ── Diagnostic logging ──────────────────────────────────────────────
        # Log the last SSO response the client processed (should be the MFA
        # verification response — the Success page with the embedded ticket).
        # This tells us whether garth correctly received the ticket from Garmin.
        try:
            lr = getattr(isolated_client, "last_resp", None)
            if lr is not None:
                print(f"[MFA] last_resp URL    : {lr.url}")
                print(f"[MFA] last_resp status : {lr.status_code}")
                # Truncate to avoid flooding logs; first 600 chars is enough
                print(f"[MFA] last_resp body   : {lr.text[:600]!r}")
        except Exception as _le:
            print(f"[MFA] could not log last_resp: {_le}")

        # If the exception carries a response (requests.HTTPError from
        # get_oauth1_token → resp.raise_for_status()), log that too.
        try:
            err_resp = getattr(e, "response", None)
            if err_resp is not None:
                print(f"[MFA] error response URL    : {err_resp.url}")
                print(f"[MFA] error response status : {err_resp.status_code}")
                print(f"[MFA] error response headers: {dict(err_resp.headers)}")
                print(f"[MFA] error response body   : {err_resp.text[:600]!r}")
        except Exception as _re:
            print(f"[MFA] could not log error response: {_re}")
        # ── End diagnostic logging ──────────────────────────────────────────

        error_str = str(e)

        # Wrong MFA code — garth 0.7.9 mobile JSON API surfaces this as
        # "SSO error: MFA_CODE_INVALID: codeEntryInvalid".  No restart needed;
        # the session is still alive and the user can try again with a fresh code.
        if "MFA_CODE_INVALID" in error_str or "codeEntryInvalid" in error_str:
            return JSONResponse(
                {
                    "error": (
                        "Invalid MFA code. "
                        "Please enter the latest 6-digit code from your authenticator app "
                        "and try again."
                    ),
                    "restart_required": False,
                },
                status_code=400,
            )

        # 401 at the MFA submission endpoint itself (verifyCode) — also a bad code,
        # but returned as a non-2xx HTTP status rather than a JSON error body.
        if "401" in error_str and "verifyCode" in error_str:
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

        # Catch-all 401 from any other endpoint.
        if "401" in error_str:
            remove_mfa_session(session_id)
            return JSONResponse(
                {
                    "error": "Authentication rejected by Garmin. Please start the setup process again.",
                    "restart_required": True,
                },
                status_code=400,
            )

        # 429 anywhere during resume_login — Cloudflare is blocking this
        # server's IP from the Garmin SSO endpoint (verifyCode or preauthorized).
        # Set the server-wide block so subsequent requests are rejected at the
        # gate rather than burning more of the rate-limit budget.
        if "429" in error_str:
            retry_after = _extract_retry_after(e)
            # Identify which endpoint was blocked for a more precise reason string.
            if "verifyCode" in error_str:
                block_reason = "429 at verifyCode — Cloudflare block on MFA endpoint"
                user_detail = (
                    "Cloudflare (Garmin's CDN) is blocking MFA authentication "
                    "requests from this server's IP address."
                )
            else:
                block_reason = "429 at OAuth preauthorized"
                user_detail = (
                    "Garmin's OAuth endpoint is rate-limiting this server's IP."
                )
            _set_garmin_block(retry_after, block_reason)
            wait_mins = max(1, (retry_after + 59) // 60)
            remove_mfa_session(session_id)
            return JSONResponse(
                {
                    "error": (
                        f"{user_detail} "
                        f"No further attempts will be made for ~{wait_mins} minute(s). "
                        f"Please try again later."
                    ),
                    "restart_required": True,
                    "rate_limited": True,
                    "retry_after": retry_after,
                },
                status_code=400,
            )

        # All other failures — session state is unknown, restart required.
        remove_mfa_session(session_id)
        return JSONResponse(
            {
                "error": f"MFA authentication failed: {e}. Please start the setup process again.",
                "restart_required": True,
            },
            status_code=400,
        )

    # Success — remove the session and proceed.
    remove_mfa_session(session_id)

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

async def api_setup_import_token(request: Request) -> JSONResponse:
    """
    Register a Garmin token obtained externally (e.g. via local_setup.py).

    Accepts a garth ``client.dumps()`` base64 string, validates it by calling
    the Garmin social-profile endpoint, then stores the token and returns the
    user's MCP URL.

    Request body: {"email": str, "token": str}

    Responses:
      200 {"mcp_url": str}   — success
      400 {"error": str}     — invalid/expired token or bad request
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

    def _validate_token(token_str: str):
        """Load token into an isolated garth client and call social profile."""
        isolated_client = GarthClient()
        isolated_client.loads(token_str)
        return isolated_client.connectapi("/userprofile-service/socialProfile")

    try:
        profile = await loop.run_in_executor(None, _validate_token, token)
    except Exception as e:
        print(f"[import-token] token validation error: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Token validation failed: {e}"},
            status_code=400,
        )

    display_name = (
        profile.get("displayName")
        or profile.get("userName")
        if isinstance(profile, dict)
        else None
    )

    try:
        mcp_url = await _save_user_and_get_url(request, token, display_name, email)
    except Exception as e:
        print(f"[import-token] save_user error: {type(e).__name__}: {e}")
        return JSONResponse(
            {"error": f"Token valid but failed to save account: {e}"},
            status_code=500,
        )

    return JSONResponse({"mcp_url": mcp_url})


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
    Route("/api/setup/import-token", api_setup_import_token, methods=["POST"]),
    Route("/api/disconnect", api_disconnect, methods=["POST"]),
]
