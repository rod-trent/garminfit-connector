"""
Multi-user Garmin adapter.

The core challenge: garth normally uses a module-level singleton (garth.client)
that writes tokens to disk. In a multi-user server every request must be isolated.

Solution: garminconnect.Garmin() creates its own garth.Client instance (garmin.garth).
When login() receives a base64 string longer than 512 chars, it calls garth.loads()
directly — pure in-memory, no disk I/O, no global state.

Per-request flow:
  1. Decrypt the user's stored garth token from DB
  2. Create a fresh Garmin() instance
  3. Call login(tokenstore=token_b64) → in-memory injection
  4. Run data calls on this isolated client
  5. If tokens were refreshed, save the new dumps() back to DB
  6. Discard the Garmin instance — nothing persists between requests
"""

import asyncio
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.auth_manager import decrypt_token, encrypt_token
from app.database import SessionLocal, User
from garmin_handler import GarminDataHandler


# ---------------------------------------------------------------------------
# Multi-user subclass of GarminDataHandler
# ---------------------------------------------------------------------------

class MultiUserGarminHandler(GarminDataHandler):
    """
    Wraps a pre-authenticated garminconnect.Garmin client.

    Inherits ALL data methods and format_data_for_context() from
    GarminDataHandler without modification. Only __init__ is overridden
    to skip file-based authentication and accept a live Garmin client.
    """

    def __init__(self, authenticated_client):
        """
        Bypass parent __init__ entirely. The parent would try to set up
        file-based garth auth — we skip that and inject the client directly.
        """
        self.client = authenticated_client
        self._authenticated = True
        self.email = getattr(authenticated_client, "username", "") or ""
        self.token_store = None
        # garmin_handler may reference self.garmin — set it as an alias
        self.garmin = authenticated_client


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def get_user_by_token(access_token: str) -> Optional[User]:
    """Look up a User record by their MCP access token."""
    async with SessionLocal() as db:
        result = await db.execute(
            select(User).where(
                User.access_token == access_token,
                User.revoked == False,  # noqa: E712
            )
        )
        return result.scalar_one_or_none()


async def update_last_used(access_token: str):
    """Update last_used_at timestamp for the user (fire-and-forget)."""
    try:
        async with SessionLocal() as db:
            result = await db.execute(
                select(User).where(User.access_token == access_token)
            )
            user = result.scalar_one_or_none()
            if user:
                user.last_used_at = datetime.utcnow()
                await db.commit()
    except Exception:
        pass  # Non-critical — don't fail tool calls over a timestamp update


async def save_refreshed_tokens(access_token: str, garmin_client) -> None:
    """
    After a Garmin API call, the garth client may have silently refreshed
    its OAuth2 token. Persist the updated token back to the database.
    """
    try:
        new_token_b64 = garmin_client.garth.dumps()
        encrypted = encrypt_token(new_token_b64)
        async with SessionLocal() as db:
            result = await db.execute(
                select(User).where(User.access_token == access_token)
            )
            user = result.scalar_one_or_none()
            if user:
                user.garth_token_encrypted = encrypted
                user.token_refreshed_at = datetime.utcnow()
                await db.commit()
    except Exception:
        pass  # Non-critical — tokens are still valid for this session


# ---------------------------------------------------------------------------
# Primary adapter function used by MCP tools
# ---------------------------------------------------------------------------

async def get_garmin_handler(access_token: str) -> "MultiUserGarminHandler":
    """
    Load the user's garth tokens from DB and return an authenticated
    MultiUserGarminHandler ready for data queries.

    Raises ValueError if the token is invalid or revoked.
    Raises RuntimeError if Garmin authentication fails.
    """
    from garminconnect import Garmin

    user = await get_user_by_token(access_token)
    if not user:
        raise ValueError(
            "Invalid or revoked access token. "
            "Please visit /setup to reconnect your Garmin account."
        )

    # Decrypt stored garth token
    token_b64 = decrypt_token(user.garth_token_encrypted)

    # Create an isolated Garmin client — its own garth.Client, no shared state
    garmin_client = Garmin()

    # login(tokenstore=base64_str) with len > 512 calls garth.loads() — in-memory only
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, garmin_client.login, token_b64)
    except Exception as e:
        raise RuntimeError(
            f"Failed to authenticate with Garmin. Your session may have expired. "
            f"Please visit /setup to reconnect. Error: {e}"
        )

    # Update last_used timestamp (fire-and-forget)
    asyncio.create_task(update_last_used(access_token))

    return MultiUserGarminHandler(garmin_client)


async def run_garmin(access_token: str, fn, *args, **kwargs):
    """
    Helper: get an authenticated handler, run a blocking Garmin call in a
    thread executor, optionally save refreshed tokens, and return the result.

    Usage in tools:
        result = await run_garmin(token, handler.get_sleep_data, date)
    """
    handler = await get_garmin_handler(access_token)
    loop = asyncio.get_event_loop()

    result = await loop.run_in_executor(None, lambda: fn(handler, *args, **kwargs))

    # Persist any token refresh that occurred during the API call
    asyncio.create_task(save_refreshed_tokens(access_token, handler.garmin))

    return result
