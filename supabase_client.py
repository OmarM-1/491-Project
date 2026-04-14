"""
supabase_client.py — Supabase Connection Layer

Two clients:
  supabase       → respects Row Level Security (use for user-scoped requests)
  supabase_admin → bypasses RLS (use for backend-only ops like memory extraction)

Environment variables required in .env:
  SUPABASE_URL              → your project URL (https://xxx.supabase.co)
  SUPABASE_ANON_KEY         → public key (safe for frontend, respects RLS)
  SUPABASE_SERVICE_ROLE_KEY → secret key (backend only, bypasses RLS)
"""

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()  # reads from .env file in project root

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY: str = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")


def _validate_config():
    """Fail fast if env vars are missing."""
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_ANON_KEY:
        missing.append("SUPABASE_ANON_KEY")
    if not SUPABASE_SERVICE_ROLE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Add them to your .env file:\n"
            f"  SUPABASE_URL=https://your-project.supabase.co\n"
            f"  SUPABASE_ANON_KEY=eyJ...\n"
            f"  SUPABASE_SERVICE_ROLE_KEY=eyJ..."
        )


# ─────────────────────────────────────────────────────────────
# Client Initialization (lazy singletons)
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_supabase():
    """
    Public client — respects Row Level Security.

    Use for any request where a user's JWT is attached.
    The JWT (from Supabase Auth) determines which rows are visible.
    
    Usage:
        from supabase_client import get_supabase
        sb = get_supabase()
        sb.auth.sign_in_with_password({"email": ..., "password": ...})
    """
    from supabase import create_client
    _validate_config()
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    logger.info(f"Supabase public client ready → {SUPABASE_URL[:40]}...")
    return client


@lru_cache(maxsize=1)
def get_supabase_admin():
    """
    Admin client — bypasses Row Level Security.

    Use ONLY for backend operations that aren't tied to a user session:
      - Extracting memories after a conversation (writing on behalf of the user)
      - Running analytics or admin queries
      - Background jobs (summarising threads, deduplicating memories)

    NEVER expose this to the frontend or pass its key to the client.

    Usage:
        from supabase_client import get_supabase_admin
        admin = get_supabase_admin()
        admin.table("user_memories").insert({...}).execute()
    """
    from supabase import create_client
    _validate_config()
    client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    logger.info("Supabase admin client ready (RLS bypassed)")
    return client


# ─────────────────────────────────────────────────────────────
# Auth helpers (wraps Supabase Auth for FastAPI)
# ─────────────────────────────────────────────────────────────

def sign_up(email: str, password: str, username: str = None):
    """
    Register a new user via Supabase Auth.
    The on_auth_user_created trigger in 001_initial_schema.sql
    automatically creates their profile row.
    """
    sb = get_supabase()
    metadata = {}
    if username:
        metadata["username"] = username
        metadata["display_name"] = username

    response = sb.auth.sign_up({
        "email": email,
        "password": password,
        "options": {"data": metadata} if metadata else {}
    })
    return response


def sign_in(email: str, password: str):
    """
    Log in and get back a session (access_token + refresh_token).
    The access_token is a JWT your frontend sends as:
        Authorization: Bearer <access_token>
    """
    sb = get_supabase()
    response = sb.auth.sign_in_with_password({
        "email": email,
        "password": password
    })
    return response


def sign_out(access_token: str):
    """Log out and invalidate the session."""
    sb = get_supabase()
    sb.auth.sign_out(access_token)


def get_user_from_token(access_token: str):
    """
    Validate a JWT and return the user object.
    Call this in your FastAPI dependency to protect endpoints.
    Returns None if the token is invalid or expired.
    """
    sb = get_supabase()
    try:
        response = sb.auth.get_user(access_token)
        return response.user
    except Exception as e:
        logger.warning(f"Token validation failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Connection test
# ─────────────────────────────────────────────────────────────

def check_connection() -> bool:
    """Verify Supabase is reachable. Useful for startup health checks."""
    try:
        admin = get_supabase_admin()
        # Simple query to confirm connectivity
        admin.table("profiles").select("id").limit(1).execute()
        return True
    except Exception as e:
        logger.error(f"Supabase connection failed: {e}")
        return False


def get_db_info() -> dict:
    """Return config info for the startup banner."""
    return {
        "backend": "Supabase (PostgreSQL)",
        "url": SUPABASE_URL[:50] + "..." if len(SUPABASE_URL) > 50 else SUPABASE_URL,
        "rls": "enabled",
        "vector": "pgvector (384-dim)",
    }
