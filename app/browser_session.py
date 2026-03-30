"""
Server-side Playwright browser sessions for Garmin authentication.

Each setup attempt gets an isolated headless Chromium browser.  Screenshots
are streamed to the user's browser over a WebSocket so they can see and
interact with the Garmin login page directly from the web UI — no local
tooling required.

Concurrency model
-----------------
Railway's smallest plan has ~512 MB RAM.  Each Chromium instance needs
~150–250 MB.  We allow at most MAX_SESSIONS concurrent sessions and
auto-close any session that has been idle for SESSION_TIMEOUT_SECS.

Session lifecycle
-----------------
1. create_session()  → browser opens, navigates to Garmin SSO → session_id
2. WebSocket streams screenshots back, forwards mouse/keyboard events
3. poll_login()      → returns cookie dict once URL leaves sso.garmin.com
4. close_session()   → browser closed, resources freed
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

MAX_SESSIONS = 3           # max concurrent Chromium browsers
SESSION_TIMEOUT_SECS = 600 # auto-close after 10 min of inactivity
SCREENSHOT_INTERVAL = 0.14 # ~7 fps

GARMIN_SSO_URL = (
    "https://sso.garmin.com/portal/sso/en-US/sign-in"
    "?clientId=GarminConnect"
    "&service=https%3A%2F%2Fconnect.garmin.com%2Fapp"
)

# module-level registry  {session_id: BrowserSession}
_sessions: dict[str, "BrowserSession"] = {}


@dataclass
class BrowserSession:
    session_id: str
    _playwright: object = field(default=None, repr=False)
    _context: object = field(default=None, repr=False)
    _page: object = field(default=None, repr=False)
    created_at: float = field(default_factory=time.monotonic)
    last_active: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_active = time.monotonic()

    @property
    def is_timed_out(self) -> bool:
        return (time.monotonic() - self.last_active) > SESSION_TIMEOUT_SECS

    async def close(self) -> None:
        _sessions.pop(self.session_id, None)
        try:
            if self._context:
                await self._context.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_session(prefill_email: str = "") -> Optional[str]:
    """
    Spin up a headless Chromium browser, navigate to Garmin SSO, and return
    a session_id.  Returns None if the server is at capacity.
    """
    # Prune timed-out sessions
    for s in [s for s in list(_sessions.values()) if s.is_timed_out]:
        log.info("Closing timed-out browser session %s", s.session_id)
        await s.close()

    if len(_sessions) >= MAX_SESSIONS:
        log.warning("Browser session limit (%d) reached", MAX_SESSIONS)
        return None

    session_id = str(uuid.uuid4())

    try:
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        context = await browser.new_context(
            viewport={"width": 1200, "height": 720},
            locale="en-US",
            timezone_id="America/New_York",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        # Mask automation signals
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )

        session = BrowserSession(
            session_id=session_id,
            _playwright=playwright,
            _context=context,
            _page=page,
        )
        _sessions[session_id] = session

        # Navigate to Garmin SSO
        await page.goto(GARMIN_SSO_URL, wait_until="domcontentloaded", timeout=30_000)

        # Pre-fill email if provided
        if prefill_email:
            try:
                email_input = page.locator('input[name="email"]').first
                await email_input.wait_for(timeout=8_000)
                await email_input.fill(prefill_email)
            except Exception:
                pass  # Not critical — user can type it themselves

        log.info("Browser session %s created", session_id)
        return session_id

    except Exception as exc:
        log.exception("Failed to create browser session: %s", exc)
        s = _sessions.pop(session_id, None)
        if s:
            await s.close()
        return None


async def get_screenshot(session_id: str) -> Optional[bytes]:
    """Return a JPEG screenshot of the current browser view."""
    session = _sessions.get(session_id)
    if not session or not session._page:
        return None
    session.touch()
    try:
        return await session._page.screenshot(
            type="jpeg", quality=70, timeout=8_000
        )
    except Exception:
        return None


async def get_current_url(session_id: str) -> str:
    session = _sessions.get(session_id)
    if not session or not session._page:
        return ""
    try:
        return session._page.url or ""
    except Exception:
        return ""


async def handle_click(session_id: str, x: float, y: float) -> None:
    session = _sessions.get(session_id)
    if session and session._page:
        session.touch()
        try:
            await session._page.mouse.click(x, y)
        except Exception:
            pass


async def handle_mouse_move(session_id: str, x: float, y: float) -> None:
    session = _sessions.get(session_id)
    if session and session._page:
        try:
            await session._page.mouse.move(x, y)
        except Exception:
            pass


async def handle_key(session_id: str, key: str) -> None:
    """Handle a keypress forwarded from the frontend canvas."""
    session = _sessions.get(session_id)
    if not session or not session._page:
        return
    session.touch()
    try:
        if len(key) == 1:
            await session._page.keyboard.type(key)
        else:
            # Named keys: Enter, Backspace, Tab, Escape, …
            await session._page.keyboard.press(key)
    except Exception:
        pass


async def handle_scroll(session_id: str, x: float, y: float, delta_y: float) -> None:
    session = _sessions.get(session_id)
    if session and session._page:
        try:
            await session._page.mouse.wheel(0, delta_y)
        except Exception:
            pass


async def poll_login(session_id: str) -> Optional[dict]:
    """
    Check whether the user has completed Garmin login.

    Returns {"cookies": {...}, "display_name": "..."} when the browser has
    landed on connect.garmin.com, None otherwise.
    """
    session = _sessions.get(session_id)
    if not session or not session._page:
        return None

    url = session._page.url
    if not url or "connect.garmin.com" not in url or "sso.garmin.com" in url:
        return None

    # Collect cookies
    try:
        raw = await session._context.cookies("https://connect.garmin.com")
        cookies = {c["name"]: c["value"] for c in raw}
    except Exception:
        return None

    if not cookies:
        return None

    # Fetch display_name while the browser has a live, authenticated session
    display_name = ""
    try:
        result = await session._page.evaluate(
            """
            async () => {
                try {
                    const r = await fetch(
                        '/gc-api/userprofile-service/socialProfile',
                        {credentials: 'include', headers: {Accept: 'application/json'}}
                    );
                    if (!r.ok) return null;
                    const d = await r.json();
                    return d.displayName || d.userName || null;
                } catch { return null; }
            }
            """
        )
        display_name = result or ""
    except Exception:
        pass

    log.info("Session %s login detected, display_name=%r", session_id, display_name)
    return {"cookies": cookies, "display_name": display_name}


def pop_session_data(session_id: str) -> Optional[dict]:
    """
    Retrieve and remove the session's cookie data without awaiting the close.
    Used by the /api/setup/complete endpoint after WebSocket confirms success.
    """
    session = _sessions.get(session_id)
    if not session:
        return None
    # Return the data if we already polled it — caller will close the session
    # We store it on the session object so complete endpoint can read it.
    return getattr(session, "_login_data", None)


def store_login_data(session_id: str, data: dict) -> None:
    """Store resolved login data on the session for the complete endpoint."""
    session = _sessions.get(session_id)
    if session:
        session._login_data = data  # type: ignore[attr-defined]


async def close_session(session_id: str) -> None:
    session = _sessions.get(session_id)
    if session:
        log.info("Closing browser session %s", session_id)
        await session.close()
