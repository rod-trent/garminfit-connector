#!/usr/bin/env python3
"""
Local Garmin authentication using Playwright.

Because Garmin's SSO is protected by Cloudflare Turnstile, the first login
must happen in a real, visible browser window on your own machine.  Once
you've logged in once, the saved browser profile lets subsequent runs work
headlessly.

Usage:
    pip install playwright httpx python-dotenv
    playwright install chrome

    # Option A — environment variables
    GARMIN_EMAIL=you@example.com GARMIN_PASSWORD=secret \
    MCP_SERVER_URL=https://your-app.railway.app \
    python scripts/playwright_setup.py

    # Option B — interactive prompts
    python scripts/playwright_setup.py

What it does:
    1. Opens a Chrome window (first run) or runs headlessly (subsequent runs)
    2. Logs in to Garmin Connect — complete MFA in the browser if prompted
    3. Extracts session cookies and your display name
    4. POSTs them to your garminfit-connector server via /api/setup/import-token
    5. Prints your personal MCP URL to use with Claude

Re-run this script whenever your Garmin session expires (~1 year).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from playwright.sync_api import BrowserContext, sync_playwright
except ImportError:
    print("playwright is not installed. Run: pip install playwright && playwright install chrome")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("httpx is not installed. Run: pip install httpx")
    sys.exit(1)

# Load .env if present (optional convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_PROFILE_DIR = Path.home() / ".garminfit-connector" / "browser_profile"

SSO_LOGIN_URL = (
    "https://sso.garmin.com/portal/sso/en-US/sign-in"
    "?clientId=GarminConnect"
    "&service=https%3A%2F%2Fconnect.garmin.com%2Fapp"
)


def _is_on_login_page(url: str) -> bool:
    return "sso.garmin.com" in url


def _has_valid_session(profile_dir: Path) -> bool:
    """Check whether a saved browser session exists (cookies file > 1 KB)."""
    cookies_file = profile_dir / "Default" / "Cookies"
    return cookies_file.exists() and cookies_file.stat().st_size > 1024


def authenticate(
    email: str,
    password: str,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    timeout_secs: int = 300,
) -> dict:
    """
    Open a browser, log in to Garmin Connect, and return the session data.

    Returns:
        {"cookies": {name: value, ...}, "display_name": "jsmith42"}
    """
    profile_dir.mkdir(parents=True, exist_ok=True)
    valid_session = _has_valid_session(profile_dir)

    if not valid_session:
        print(
            "\nFirst login: a Chrome window will open.\n"
            "Complete the login (and any MFA) in the browser.\n"
            "Subsequent runs will be fully headless.\n"
        )

    launch_args = ["--disable-blink-features=AutomationControlled"]
    if valid_session:
        launch_args.append("--headless=new")

    with sync_playwright() as p:
        context: BrowserContext = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,        # overridden by --headless=new in args when needed
            channel="chrome",
            args=launch_args,
            ignore_default_args=["--enable-automation"],
            locale="en-US",
            timezone_id="America/New_York",
        )

        page = context.pages[0] if context.pages else context.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        )

        # Navigate to the Garmin Connect app
        try:
            page.goto(
                "https://connect.garmin.com/modern/",
                wait_until="domcontentloaded",
            )
        except Exception:
            pass
        time.sleep(3)

        current_url = page.url

        if _is_on_login_page(current_url):
            print("Logging in…")

            # Clear stale cookies to avoid redirect loops
            try:
                context.clear_cookies()
            except Exception:
                pass

            for attempt in range(3):
                try:
                    page.goto(SSO_LOGIN_URL, wait_until="domcontentloaded")
                    break
                except Exception:
                    time.sleep(3)

            time.sleep(2)

            # Fill credentials
            try:
                email_input = page.locator('input[name="email"]').first
                email_input.wait_for(timeout=20_000)
                email_input.click()
                page.keyboard.type(email, delay=30)

                pwd_input = page.locator('input[name="password"]').first
                pwd_input.wait_for(timeout=5_000)
                pwd_input.click()
                page.keyboard.type(password, delay=30)

                submit = page.locator('button[type="submit"]').first
                submit.click()
                print("Credentials submitted — waiting for Garmin…")
            except Exception as exc:
                print(f"Could not fill login form: {exc}")
                print("Try completing the login manually in the browser window.")

            # Wait for redirect away from SSO (user completes MFA in browser)
            for tick in range(timeout_secs):
                time.sleep(1)
                url = page.url

                # Check if any tab landed on connect.garmin.com (not SSO)
                for p_tab in context.pages:
                    if "connect.garmin.com" in p_tab.url and "sso.garmin.com" not in p_tab.url:
                        page = p_tab
                        url = p_tab.url
                        break

                if "connect.garmin.com" in url and "sso.garmin.com" not in url:
                    break

                if tick % 30 == 29:
                    print(f"Still waiting for login… ({tick + 1}s)")
            else:
                context.close()
                raise TimeoutError(
                    f"Login did not complete within {timeout_secs}s. "
                    "Make sure you completed any MFA challenge in the browser."
                )

            print("Login successful!")
        else:
            print("Session restored from saved profile.")

        # Fetch display_name via the Garmin Connect social-profile API
        display_name = _fetch_display_name(page) or email.split("@")[0]

        # Collect all Garmin Connect cookies
        all_cookies = context.cookies("https://connect.garmin.com")
        cookies_dict = {c["name"]: c["value"] for c in all_cookies}

        context.close()

    return {"cookies": cookies_dict, "display_name": display_name}


def _fetch_display_name(page) -> Optional[str]:
    """Ask the Garmin API for the display name while the browser is live."""
    try:
        result = page.evaluate(
            """
            async () => {
                try {
                    const resp = await fetch(
                        '/gc-api/userprofile-service/socialProfile',
                        {credentials: 'include', headers: {Accept: 'application/json'}}
                    );
                    if (!resp.ok) return null;
                    const data = await resp.json();
                    return data.displayName || data.userName || null;
                } catch { return null; }
            }
            """
        )
        return result
    except Exception:
        return None


def import_to_server(server_url: str, email: str, session_data: dict) -> str:
    """
    POST session cookies to the garminfit-connector server.

    Returns the MCP URL for this user.
    """
    token_json = json.dumps(session_data)
    url = f"{server_url.rstrip('/')}/api/setup/import-token"

    resp = httpx.post(
        url,
        json={"email": email, "token": token_json},
        timeout=30,
    )

    if not resp.is_success:
        try:
            err = resp.json().get("error", resp.text)
        except Exception:
            err = resp.text
        print(f"ERROR: Server returned {resp.status_code}: {err}")
        sys.exit(1)

    data = resp.json()

    if "mcp_url" not in data:
        raise ValueError(f"Unexpected server response: {data}")

    return data["mcp_url"]


def main() -> None:
    email = os.environ.get("GARMIN_EMAIL") or input("Garmin email: ").strip()
    password = os.environ.get("GARMIN_PASSWORD") or input("Garmin password: ").strip()
    server_url = (
        os.environ.get("MCP_SERVER_URL")
        or input("Garminfit-connector server URL (e.g. https://your-app.railway.app): ").strip()
    )

    if not email or not password or not server_url:
        print("Email, password, and server URL are all required.")
        sys.exit(1)

    print("\nAuthenticating with Garmin…")
    session_data = authenticate(email, password)
    print(f"Authenticated as: {session_data['display_name']}")

    print("\nUploading session to server…")
    mcp_url = import_to_server(server_url, email, session_data)

    print(f"\n✅ Done!\n\nYour MCP URL:\n  {mcp_url}\n")
    print(
        "Add this URL to your Claude Desktop config or any MCP-compatible client.\n"
        "Run this script again if your session expires (~1 year)."
    )


if __name__ == "__main__":
    main()
