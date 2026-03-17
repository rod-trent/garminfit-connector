"""
Authentication utilities:
  - Fernet encryption/decryption for garth tokens stored in DB
  - In-memory MFA pending session store (5-minute TTL)
  - Access token generation for user MCP URLs

MFA design (v2 — garth return_on_mfa)
--------------------------------------
garth 0.6+ exposes sso.login(return_on_mfa=True) which returns immediately with
a client_state dict when MFA is required, instead of blocking inside a callback.
sso.resume_login(client_state, mfa_code) then completes the login in a second
call.  This removes the need for a blocking thread bridge entirely, which was
the root cause of the 401 at the OAuth preauthorized endpoint — the old bridge
kept the SSO session open while the user typed, and the CAS service ticket
could expire before the OAuth exchange had a chance to run.

The PendingMFASession now stores just the garth client_state dict and the
isolated garth.http.Client so resume_login can complete the flow.

NOTE: Pending MFA sessions are stored in-memory on a single Railway dyno.
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
    Generate a cryptographically random URL-safe token (~22 chars).
    This is the token embedded in the user's MCP URL path:
      https://garminfit-connector.railway.app/garmin/?token={access_token}

    token_urlsafe(16) produces a base64url string (a-z A-Z 0-9 - _)
    that is shorter and less likely to be flagged by WAF rules than a
    64-character hex string (which resembles a SHA-256 hash).
    """
    return secrets.token_urlsafe(16)  # 16 bytes → ~22 URL-safe chars


# ---------------------------------------------------------------------------
# In-memory MFA pending session store
# ---------------------------------------------------------------------------

@dataclass
class PendingMFASession:
    """
    Holds everything needed to complete a pending MFA login using garth's
    native return_on_mfa / resume_login API.

    client_state: the dict returned by sso.login(return_on_mfa=True) when
        MFA is required.  Contains {"signin_params": ..., "client": ...}.
        The embedded "client" key IS isolated_client — same object.

    isolated_client: the garth.http.Client instance used for this login.
        Stored separately so we can call isolated_client.configure() and
        isolated_client.dumps() after resume_login() returns the tokens.
    """
    session_id: str
    garmin_email: str
    client_state: dict        # returned by sso.login(return_on_mfa=True)
    isolated_client: object   # garth.http.Client instance (for configure + dumps)
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(minutes=5)
    )


# Module-level dict: session_id → PendingMFASession
_pending: dict[str, PendingMFASession] = {}

MFA_TTL_MINUTES = 5


def create_mfa_session(
    garmin_email: str,
    client_state: dict,
    isolated_client: object,
) -> str:
    """
    Store a pending MFA session and return its session_id.
    The caller should return session_id to the setup page JS.
    """
    _cleanup_expired()
    session_id = secrets.token_urlsafe(24)
    _pending[session_id] = PendingMFASession(
        session_id=session_id,
        garmin_email=garmin_email,
        client_state=client_state,
        isolated_client=isolated_client,
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
