"""
Web routes for the user-facing setup and disconnect flows.

Auth model (server-side browser session):
------------------------------------------
Garmin's SSO is protected by Cloudflare Turnstile.  We run a headless
Chromium browser on the server (via Playwright) and stream it to the user's
browser as an interactive view.  The user sees and controls the Garmin login
page directly — no local tooling required.

Setup flow
----------
1. GET  /setup           — setup page with email input + "Connect" button
2. WS   /api/setup/ws    — opens WebSocket; server creates Chromium session,
                           streams JPEG screenshots (binary frames) and JSON
                           control messages (text frames); client forwards
                           mouse/keyboard events as JSON text frames
3. POST /api/setup/complete — called by JS after WS signals login_success;
                              saves cookies → returns {"mcp_url": ...}
4. POST /api/disconnect   — revoke by email

Other routes
------------
GET  /            → redirect to /setup
GET  /disconnect  → disconnect form
GET  /health      → Railway health check
GET  /debug/mcp   → MCP session diagnostics
"""

import asyncio
import json
import os
from datetime import datetime

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from app.auth_manager import encrypt_token, generate_access_token
from app.browser_session import (
    close_session,
    create_session,
    get_screenshot,
    get_current_url,
    handle_click,
    handle_key,
    handle_mouse_move,
    handle_scroll,
    poll_login,
    pop_session_data,
    store_login_data,
)
from app.database import SessionLocal, User
from app.garmin_api_client import GarminApiClient


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
# WebSocket: stream browser session to user
# ---------------------------------------------------------------------------

async def browser_ws(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that drives a server-side Chromium browser.

    Binary frames  → JPEG screenshots sent to the client (~7 fps)
    Text frames    → JSON control messages to/from client

    Client → server text events:
      {"type": "click",      "x": float, "y": float}
      {"type": "mousemove",  "x": float, "y": float}
      {"type": "key",        "key": str}
      {"type": "scroll",     "x": float, "y": float, "deltaY": float}
      {"type": "start",      "email": str}   ← first message from client

    Server → client text events:
      {"type": "ready"}
      {"type": "url",          "url": str}
      {"type": "login_success","session_id": str}
      {"type": "error",        "message": str}
    """
    await websocket.accept()

    session_id: str | None = None

    # ---- wait for the start message with (optional) email
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=15)
        msg = json.loads(raw)
        if msg.get("type") != "start":
            await websocket.send_text(json.dumps({"type": "error", "message": "Expected start message"}))
            await websocket.close()
            return
        email = (msg.get("email") or "").strip()
    except (asyncio.TimeoutError, Exception) as exc:
        await websocket.send_text(json.dumps({"type": "error", "message": f"Setup error: {exc}"}))
        await websocket.close()
        return

    # ---- create the browser session
    session_id = await create_session(prefill_email=email)
    if not session_id:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Server is busy — all browser slots are in use. Please try again in a minute.",
        }))
        await websocket.close()
        return

    await websocket.send_text(json.dumps({"type": "ready"}))

    # ---- concurrent tasks: receive events + stream screenshots
    login_result: dict | None = None
    stop_event = asyncio.Event()

    async def recv_task() -> None:
        """Read mouse/keyboard events from the client."""
        try:
            while not stop_event.is_set():
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                event = json.loads(raw)
                etype = event.get("type")
                if etype == "click":
                    await handle_click(session_id, event["x"], event["y"])
                elif etype == "mousemove":
                    await handle_mouse_move(session_id, event["x"], event["y"])
                elif etype == "key":
                    await handle_key(session_id, event["key"])
                elif etype == "scroll":
                    await handle_scroll(session_id, event["x"], event["y"], event.get("deltaY", 0))
        except (WebSocketDisconnect, Exception):
            stop_event.set()

    async def stream_task() -> None:
        """Send screenshots and check for login success."""
        nonlocal login_result
        try:
            while not stop_event.is_set():
                # Check for login success first (cheaper than screenshot)
                result = await poll_login(session_id)
                if result:
                    login_result = result
                    store_login_data(session_id, result)
                    await websocket.send_text(json.dumps({
                        "type": "login_success",
                        "session_id": session_id,
                    }))
                    stop_event.set()
                    return

                # Stream screenshot
                screenshot = await get_screenshot(session_id)
                if screenshot and websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_bytes(screenshot)
                    except Exception:
                        stop_event.set()
                        return

                # Send current URL so the client can show a status bar
                url = await get_current_url(session_id)
                if websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_text(json.dumps({"type": "url", "url": url}))
                    except Exception:
                        stop_event.set()
                        return

                await asyncio.sleep(0.14)  # ~7 fps
        except Exception:
            stop_event.set()

    try:
        await asyncio.gather(recv_task(), stream_task())
    finally:
        # Don't close the session yet — /api/setup/complete still needs it
        if not login_result:
            await close_session(session_id)


# ---------------------------------------------------------------------------
# POST /api/setup/complete  — save cookies after WebSocket login_success
# ---------------------------------------------------------------------------

async def api_setup_complete(request: Request) -> JSONResponse:
    """
    Called by the frontend after the WebSocket signals login_success.

    Request body: {"session_id": str, "email": str}
    Response:     {"mcp_url": str}
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    session_id = (body.get("session_id") or "").strip()
    email = (body.get("email") or "").strip()

    if not session_id:
        return JSONResponse({"error": "session_id is required"}, status_code=400)

    # Retrieve the login data stored by the WebSocket stream task
    login_data = pop_session_data(session_id)
    if not login_data:
        return JSONResponse(
            {"error": "Session not found or login not completed. Please try again."},
            status_code=404,
        )

    cookies = login_data.get("cookies", {})
    display_name = login_data.get("display_name", "")

    # If email wasn't in the form, derive it from display_name as a fallback
    if not email:
        email = f"{display_name}@garmin" if display_name else "unknown@garmin"

    # Persist updated display_name in the token
    token_json = GarminApiClient(cookies=cookies, display_name=display_name).dumps()

    try:
        mcp_url = await _save_user_and_get_url(request, token_json, display_name, email)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to save account: {exc}"}, status_code=500)
    finally:
        # Always close the browser once we're done with it
        await close_session(session_id)

    return JSONResponse({"mcp_url": mcp_url})


# ---------------------------------------------------------------------------
# POST /api/setup/import-token  — for the playwright_setup.py script (kept)
# ---------------------------------------------------------------------------

async def api_setup_import_token(request: Request) -> JSONResponse:
    """
    Register a Garmin session obtained externally (e.g. scripts/playwright_setup.py).

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
        data = client._get("/userprofile-service/socialProfile")
        display_name = (
            data.get("displayName") or data.get("userName")
            if isinstance(data, dict)
            else None
        )
        if display_name:
            client.display_name = display_name
        return display_name, client.dumps()

    try:
        display_name, updated_token = await loop.run_in_executor(None, _validate, token)
    except Exception as exc:
        return JSONResponse({"error": f"Token validation failed: {exc}"}, status_code=400)

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
# Route list (imported by main.py)
# ---------------------------------------------------------------------------

setup_routes = [
    Route("/", root, methods=["GET"]),
    Route("/setup", setup_page, methods=["GET"]),
    Route("/setup/success", setup_success_page, methods=["GET"]),
    Route("/disconnect", disconnect_page, methods=["GET"]),
    Route("/health", health_check, methods=["GET"]),
    Route("/debug/mcp", debug_mcp, methods=["GET"]),
    WebSocketRoute("/api/setup/ws", browser_ws),
    Route("/api/setup/complete", api_setup_complete, methods=["POST"]),
    Route("/api/setup/import-token", api_setup_import_token, methods=["POST"]),
    Route("/api/disconnect", api_disconnect, methods=["POST"]),
]
