"""
SeleniumBase UC-based Garmin authentication (replaces browser_session.py).

Each login attempt runs a SeleniumBase UC browser in a background thread.
The thread signals state changes via threading.Event so the async HTTP
handler can await them without blocking the event loop.

Session states:
  pending → mfa_required → success
  pending → success
  pending → error

UC mode patches the ChromeDriver binary itself (not via JS overrides), so
Cloudflare passes without a residential proxy. On headless Linux (Railway)
we use xvfb=True which spawns a virtual X display — Chrome's built-in
headless mode fails Cloudflare fingerprint checks.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

GARMIN_CONNECT_URL = "https://connect.garmin.com/modern/"
STATE_CHANGE_TIMEOUT = 120   # seconds to wait for login form / redirect
MFA_INPUT_TIMEOUT    = 300   # seconds to wait for user to supply MFA code
SESSION_MAX_AGE_SECS = 660   # prune sessions older than this


@dataclass
class UCLoginSession:
    session_id: str
    email: str
    _password: str
    state: str = "pending"          # pending | mfa_required | success | error
    result: Optional[dict] = None   # {"cookies": {...}, "display_name": "..."}
    error: Optional[str] = None
    _mfa_code: Optional[str] = None
    _state_changed: threading.Event = field(default_factory=threading.Event)
    _mfa_ready: threading.Event = field(default_factory=threading.Event)
    created_at: float = field(default_factory=time.monotonic)

    def wait_for_state_change(self, timeout: float = STATE_CHANGE_TIMEOUT) -> str:
        """Block until state leaves 'pending'. Returns the new state."""
        self._state_changed.wait(timeout=timeout)
        self._state_changed.clear()
        return self.state

    def submit_mfa(self, code: str) -> None:
        """Called from the HTTP handler to unblock the waiting login thread."""
        self._mfa_code = code
        self._mfa_ready.set()

    def _transition(self, state: str) -> None:
        self.state = state
        self._state_changed.set()

    def run(self) -> None:
        try:
            self._do_login()
        except Exception as exc:
            log.exception("UC login failed for session %s", self.session_id[:8])
            self.error = str(exc)
            self._transition("error")

    def _do_login(self) -> None:
        import os
        from seleniumbase import SB

        # On headless Linux (Railway etc.) xvfb=True spawns a virtual X display.
        # UC mode is unreliable with Chrome's --headless flag.
        use_xvfb = not bool(os.environ.get("DISPLAY", ""))

        # Explicitly locate the Chrome/Chromium binary so SeleniumBase doesn't
        # fall back to a slow/failing auto-search inside the container.
        # Railway nixpkgs installs chromium to the nix profile bin directory.
        _CHROME_CANDIDATES = [
            "/root/.nix-profile/bin/chromium",   # Railway nixpkgs
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/usr/bin/google-chrome-stable",      # Dockerfile / apt install
            "/usr/bin/google-chrome",
        ]
        binary = next((p for p in _CHROME_CANDIDATES if os.path.exists(p)), None)
        sb_kwargs = dict(uc=True, headless=False, xvfb=use_xvfb)
        if binary:
            sb_kwargs["binary_location"] = binary
            log.info("Using Chrome binary: %s", binary)
        else:
            log.warning("No Chrome binary found at known paths; SeleniumBase will search PATH")

        with SB(**sb_kwargs) as sb:
            # uc_open_with_reconnect briefly disconnects CDP during navigation
            # so Cloudflare sees the request as human-initiated.
            sb.uc_open_with_reconnect(GARMIN_CONNECT_URL, reconnect_time=6)

            # Dismiss any Cloudflare CAPTCHA if it appears
            try:
                sb.uc_gui_click_captcha()
            except Exception:
                pass

            # Wait for the Garmin SSO login form
            try:
                sb.wait_for_element('input[name="email"]', timeout=40)
            except Exception:
                self.error = "Garmin login page did not load — Cloudflare may have blocked the request"
                self._transition("error")
                return

            # Type credentials (SeleniumBase types with human-like cadence by default)
            sb.type('input[name="email"]', self.email)
            sb.type('input[name="password"]', self._password)

            # Check "Remember me" to extend the session lifetime (~30 days)
            try:
                sb.click('input[id="rememberMe"]', timeout=2)
            except Exception:
                pass

            sb.click('button[type="submit"]')
            sb.sleep(3)

            # ── MFA detection ─────────────────────────────────────────────
            current_url = sb.get_current_url()
            if any(x in current_url.lower() for x in ("mfa", "verif", "security-code", "totp")):
                self._transition("mfa_required")

                if not self._mfa_ready.wait(timeout=MFA_INPUT_TIMEOUT):
                    self.error = "MFA timeout — no code received"
                    self._transition("error")
                    return

                _MFA_SELECTORS = [
                    'input[name="verificationCode"]',
                    'input[name="securityCode"]',
                    'input[type="tel"]',
                    'input[autocomplete="one-time-code"]',
                    'input[placeholder*="code"]',
                ]
                filled = False
                for sel in _MFA_SELECTORS:
                    try:
                        sb.wait_for_element(sel, timeout=4)
                        sb.type(sel, self._mfa_code)
                        filled = True
                        break
                    except Exception:
                        continue

                if not filled:
                    self.error = "Could not find MFA input field on the page"
                    self._transition("error")
                    return

                sb.click('button[type="submit"]')
                sb.sleep(3)

            # ── Wait for successful redirect to Garmin Connect ────────────
            for _ in range(30):
                url = sb.get_current_url()
                if "connect.garmin.com" in url and "sso.garmin.com" not in url:
                    break
                sb.sleep(1)
            else:
                self.error = "Login timed out — never reached Garmin Connect after credentials"
                self._transition("error")
                return

            # ── Extract Garmin cookies ────────────────────────────────────
            cookies = {
                c["name"]: c["value"]
                for c in sb.get_cookies()
                if "garmin" in c.get("domain", "")
            }

            if not cookies:
                self.error = "Login appeared successful but no Garmin cookies were found"
                self._transition("error")
                return

            # ── Fetch display name via authenticated browser fetch() ──────
            display_name = ""
            try:
                display_name = sb.execute_async_script("""
                    var done = arguments[arguments.length - 1];
                    fetch('/gc-api/userprofile-service/socialProfile', {
                        credentials: 'include',
                        headers: {Accept: 'application/json'}
                    })
                    .then(r => r.json())
                    .then(d => done(d.displayName || d.userName || ''))
                    .catch(() => done(''));
                """) or ""
            except Exception:
                pass

            self.result = {"cookies": cookies, "display_name": display_name}
            self._transition("success")


# ── Session registry ──────────────────────────────────────────────────────────

_sessions: dict[str, UCLoginSession] = {}
_lock = threading.Lock()


def create_uc_session(email: str, password: str) -> UCLoginSession:
    """Create a login session and start it in a background thread."""
    _prune_sessions()
    session = UCLoginSession(
        session_id=str(uuid.uuid4()),
        email=email,
        _password=password,
    )
    with _lock:
        _sessions[session.session_id] = session
    threading.Thread(
        target=session.run,
        daemon=True,
        name=f"uc-login-{session.session_id[:8]}",
    ).start()
    log.info("UC login session %s started for %s", session.session_id[:8], email)
    return session


def get_uc_session(session_id: str) -> Optional[UCLoginSession]:
    with _lock:
        return _sessions.get(session_id)


def remove_uc_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


def _prune_sessions() -> None:
    now = time.monotonic()
    with _lock:
        stale = [
            sid for sid, s in _sessions.items()
            if s.state in ("success", "error")
            or (now - s.created_at) > SESSION_MAX_AGE_SECS
        ]
        for sid in stale:
            del _sessions[sid]
