#!/usr/bin/env python3
"""
Garmin Chat Connector — Local Setup Script
==========================================
Run this on your own computer to authenticate with Garmin
and register your MCP URL with the Garmin Chat Connector.

Why local? Garmin's OAuth endpoints are protected by Cloudflare
which blocks requests from cloud/data-centre IPs like Railway.
Running locally uses your residential IP which Cloudflare allows.

Usage:
    python local_setup.py
    python local_setup.py --app-url https://your-app.railway.app
"""

import argparse
import getpass
import sys

APP_URL_DEFAULT = "https://garminfit-connector-production.up.railway.app"

def main():
    parser = argparse.ArgumentParser(description="Local Garmin Chat Connector setup")
    parser.add_argument("--app-url", default=APP_URL_DEFAULT,
                        help=f"Your Railway app URL (default: {APP_URL_DEFAULT})")
    args = parser.parse_args()

    # Lazy-import so the script gives a helpful error if deps are missing
    try:
        import garth
    except ImportError:
        print("ERROR: garth is not installed.")
        print("Run:  pip install garth")
        sys.exit(1)
    try:
        import requests as _req
    except ImportError:
        print("ERROR: requests is not installed.")
        print("Run:  pip install requests")
        sys.exit(1)

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
    print("Connecting to Garmin…  (enter your MFA code if prompted)")

    client = garth.Client()
    try:
        client.login(email, password)
    except Exception as exc:
        print(f"\nERROR: Garmin login failed: {exc}")
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
