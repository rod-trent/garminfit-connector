#!/usr/bin/env python3
"""
Garmin Chat Connector — Local Setup Script
==========================================
Run this on your own computer to authenticate with Garmin
and register your MCP URL with the Garmin Chat Connector.

Why local?  Running the Garmin OAuth flow locally can be more reliable than
going through the web server, particularly for troubleshooting.

Usage:
    python local_setup.py
    python local_setup.py --app-url https://your-app.railway.app
    python local_setup.py --debug          (verbose output for troubleshooting)
"""

import argparse
import getpass
import sys

APP_URL_DEFAULT = "https://garminfit-connector-production.up.railway.app"


def main():
    parser = argparse.ArgumentParser(description="Local Garmin Chat Connector setup")
    parser.add_argument(
        "--app-url",
        default=APP_URL_DEFAULT,
        help=f"Your Railway app URL (default: {APP_URL_DEFAULT})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print verbose diagnostic output (useful for troubleshooting)",
    )
    args = parser.parse_args()

    # Lazy-import so the script gives a helpful error if deps are missing
    try:
        import garth
        import garth.sso as garth_sso
    except ImportError:
        print("ERROR: garth is not installed.")
        print("Run:  pip install 'garth>=0.7.10'")
        sys.exit(1)

    # Patch garth 0.7.10: retry preauthorized on 401/429 (garth PR #214)
    import time as _time
    _orig_get_oauth1_token = garth_sso.get_oauth1_token

    def _patched_get_oauth1_token(ticket, client, retries=3):
        retries = max(retries, 1)
        last_exc = None
        for attempt in range(retries):
            try:
                return _orig_get_oauth1_token(ticket, client)
            except Exception as exc:
                err = str(exc)
                if attempt < retries - 1 and ("401" in err or "429" in err):
                    wait = 1 * (attempt + 1)
                    dbg(f"preauth attempt {attempt + 1} failed ({exc!s:.80}); retrying in {wait}s …")
                    _time.sleep(wait)
                    last_exc = exc
                    continue
                raise
        raise last_exc

    garth_sso.get_oauth1_token = _patched_get_oauth1_token

    # Debug _complete_login: try both service URLs at the preauthorized step so
    # we can see exactly which one Garmin accepts (garth uses the mobile URL by
    # default; connect.garmin.com/modern/ is the web fallback).
    _CLASSIC_SERVICE = "https://connect.garmin.com/modern/"

    def _debug_complete_login(ticket, client):
        import requests as _requests
        from urllib.parse import parse_qs as _parse_qs
        from garth.auth_tokens import OAuth1Token as _OAuth1Token
        dbg(f"_complete_login: ticket={ticket[:30]}…")
        # OAUTH_CONSUMER is populated lazily; log after first GarminOAuth1Session call
        base_url = f"https://connectapi.{client.domain}/oauth-service/oauth/"

        # Try mobile URL first (garth 0.7.10 issues tickets for this), then
        # connect.garmin.com/modern/ as fallback in case the mobile URL is rejected.
        mobile_url = f"https://mobile.integration.{client.domain}/gcm/android"
        for login_url in [mobile_url, _CLASSIC_SERVICE]:
            url = (f"{base_url}preauthorized?ticket={ticket}"
                   f"&login-url={login_url}&accepts-mfa-tokens=true")
            dbg(f"trying login-url={login_url!r}")
            for label, sess in [
                ("with-cookies", garth_sso.GarminOAuth1Session(parent=client.sess)),
                ("clean",        garth_sso.GarminOAuth1Session()),
            ]:
                full_label = f"{login_url.split('/')[-2]}/{label}"
                dbg(f"[{full_label}] consumer_key={garth_sso.OAUTH_CONSUMER.get('consumer_key','?')[:8]}…")
                try:
                    resp = sess.get(url, headers=garth_sso.OAUTH_USER_AGENT, timeout=client.timeout)
                    dbg(f"[{full_label}] HTTP {resp.status_code}")
                    if not resp.ok:
                        dbg(f"[{full_label}] body: {resp.text[:300]!r}")
                    resp.raise_for_status()
                    parsed = _parse_qs(resp.text)
                    token = {k: v[0] for k, v in parsed.items()}
                    oauth1 = _OAuth1Token(domain=client.domain, **token)
                    dbg(f"[{full_label}] SUCCESS!")
                    oauth2 = garth_sso.exchange(oauth1, client, login=True)
                    return oauth1, oauth2
                except Exception as exc:
                    dbg(f"[{full_label}] FAILED: {exc!s:.160}")

        raise Exception("All preauthorized attempts failed — see debug output above")

    garth_sso._complete_login = _debug_complete_login

    try:
        import requests as _req
    except ImportError:
        print("ERROR: requests is not installed.")
        print("Run:  pip install requests")
        sys.exit(1)

    def dbg(msg):
        if args.debug:
            print(f"  [debug] {msg}")

    print()
    print("Garmin Chat Connector — Local Setup")
    print("=" * 42)
    print("Authenticate with Garmin on this machine,")
    print("then register the token with your app.")
    print()

    email = input("Garmin Connect email: ").strip()
    if not email:
        print("ERROR: email is required.")
        sys.exit(1)
    password = getpass.getpass("Garmin Connect password: ")
    if not password:
        print("ERROR: password is required.")
        sys.exit(1)

    print()
    print("Connecting to Garmin…")

    def _prompt_mfa():
        print()
        print("🔐  MFA required — check your authenticator app.")
        mfa_code = input("   Enter your 6-digit MFA code: ").strip()
        if not mfa_code:
            print("ERROR: MFA code cannot be empty.")
            sys.exit(1)
        dbg(f"MFA code entered (len={len(mfa_code)})")
        return mfa_code

    client = garth.Client()
    try:
        dbg("Calling garth sso.login() with blocking prompt_mfa ...")
        oauth1, oauth2 = garth_sso.login(
            email, password, client=client, prompt_mfa=_prompt_mfa
        )
        dbg("OAuth tokens obtained.")

        client.configure(
            oauth1_token=oauth1,
            oauth2_token=oauth2,
            domain=oauth1.domain,
        )

    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: Garmin login failed: {exc}")
        if not args.debug:
            print("Tip: re-run with --debug for more detail.")
        sys.exit(1)

    token = client.dumps()
    print("✓ Garmin authentication successful.")
    print()
    print(f"Registering token with {args.app_url} …")

    try:
        resp = _req.post(
            f"{args.app_url}/api/setup/import-token",
            json={"email": email, "token": token},
            timeout=30,
        )
    except Exception as exc:
        print(f"\nERROR: Could not reach app: {exc}")
        sys.exit(1)

    if resp.status_code == 200:
        mcp_url = resp.json().get("mcp_url", "(not returned)")
        print()
        print("✅  Setup complete!")
        print()
        print("Your MCP URL:")
        print()
        print(f"   {mcp_url}")
        print()
        print("Add this URL to Claude (or another AI tool) as an MCP server.")
    else:
        try:
            err = resp.json().get("error", resp.text)
        except Exception:
            err = resp.text
        print(f"\nERROR: App returned {resp.status_code}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
