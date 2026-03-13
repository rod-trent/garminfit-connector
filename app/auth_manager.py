"""
Authentication utilities:
  - Fernet encryption/decryption for garth tokens stored in DB
  - In-memory MFA pending session store (5-minute TTL)
  - Access token generation for user MCP URLs

NOTE: The pending MFA sessions are stored in-memory on a single Railway dyno.
If the server restarts during a user's 5-minute MFA window, their session is
lost and they must restart the setup process. This is acceptable for V1.
For multi-dyno deployments, replace _pending with a Redis-backed store.
"""

import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from cryptography.fernet import Fernet

# ---------------------------------------------------------------------------
# Token encryption
# ---------------------------------------------------------------------------

def _get_fernet() -> Fernet:
    key = os.environ.get("TOKEN_ENCRYPTION_KEY", "")
    if not key:
        raise RuntimeError(
            "TOKEN_ENCRYPTION_KEY environment variable is not set. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    return Fernet(key.encode())


def encrypt_token(token_b64: str) -> str:
    """Encrypt a garth dumps() base64 string for storage in the database."""
    return _get_fernet().encrypt(token_b64.encode()).decode()


def decrypt_token(encrypted: str) -> str:
    """Decrypt a stored garth token back to a base64 string for garth.loads()."""
    return _get_fernet().decrypt(encrypted.encode()).decode()


# ---------------------------------------------------------------------------
# Access token generation (goes in the user's MCP URL)
# ---------------------------------------------------------------------------

def generate_access_token() -> str:
    """
    Generate a cryptographically random 64-character hex access token.
    This is the token embedded in the user's MCP URL path:
      https://garminfit-connector.railway.app/mcp/{access_token}/sse
    """
    return secrets.token_hex(32)  # 32 bytes → 64 hex characters


# ---------------------------------------------------------------------------
# In-memory MFA pending session store
# ---------------------------------------------------------------------------

@dataclass
class PendingMFASession:
    """
    Holds the live garminconnect.Garmin instance mid-MFA.

    The garmin_client has an active garth.Client with live Garmin SSO session
    cookies that are required to complete MFA. These cannot be serialized to
    the database, so they live here in memory for up to 5 minutes.
    """
    session_id: str
    garmin_client: object          # garminconnect.Garmin instance
    client_state: dict             # returned by garth on MFA — contains live Client
    garmin_email: str
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=5))


# Module-level dict: session_id → PendingMFASession
_pending: dict[str, PendingMFASession] = {}

MFA_TTL_MINUTES = 5


def create_mfa_session(garmin_client: object, client_state: dict, email: str) -> str:
    """
    Store a pending MFA session and return its session_id.
    The caller should return session_id to the setup page JS.
    """
    _cleanup_expired()
    session_id = secrets.token_urlsafe(24)
    _pending[session_id] = PendingMFASession(
        session_id=session_id,
        garmin_client=garmin_client,
        client_state=client_state,
        garmin_email=email,
    )
    return session_id


def pop_mfa_session(session_id: str) -> PendingMFASession | None:
    """
    Retrieve and remove a pending MFA session.
    Returns None if the session doesn't exist or has expired.
    """
    _cleanup_expired()
    session = _pending.pop(session_id, None)
    if session is None:
        return None
    if datetime.utcnow() > session.expires_at:
        return None  # expired but wasn't cleaned up yet
    return session


def _cleanup_expired():
    """Remove expired sessions from the in-memory store."""
    now = datetime.utcnow()
    expired = [sid for sid, s in _pending.items() if now > s.expires_at]
    for sid in expired:
        del _pending[sid]
