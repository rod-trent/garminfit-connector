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
        print("Run:  pip install 'garth>=0.7.9'")
        sys.exit(1)
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

    client = garth.Client()
    try:
        dbg("Calling garth sso.login(return_on_mfa=True) ...")
        result = garth_sso.login(email, password, client=client, return_on_mfa=True)

        if isinstance(result, tuple) and result[0] == "needs_mfa":
            _, client_state = result
            print()
            print("🔐  MFA required — check your authenticator app.")
            mfa_code = input("   Enter your 6-digit MFA code: ").strip()
            if not mfa_code:
                print("ERROR: MFA code cannot be empty.")
                sys.exit(1)

            dbg("Calling garth sso.resume_login() ...")
            oauth1, oauth2 = garth_sso.resume_login(client_state, mfa_code)
            dbg("OAuth tokens obtained via resume_login.")
        else:
            oauth1, oauth2 = result
            dbg("No MFA required — login completed by garth directly.")

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
