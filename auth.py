"""
Auth layer — Google OAuth only.
- No passwords stored anywhere
- Google verifies the email actually exists and belongs to the user
- Auto-approve logic: execfunctions.co domain (or any APPROVED_DOMAINS) + admin whitelist
- Sessions stored server-side in PostgreSQL (shared pool from database.py)
"""
import os
import secrets
from datetime import datetime, timedelta
import psycopg2.extras

from database import get_conn, return_conn

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "").lower().strip()
APPROVED_DOMAINS = [d.lower().strip() for d in os.environ.get("APPROVED_DOMAINS", "execfunctions.co").split(",") if d.strip()]
SESSION_TTL_HOURS = 720  # 30 days


def init_auth_db():
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                picture TEXT,
                is_approved INTEGER DEFAULT 0,
                is_admin INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                approved_at TEXT,
                last_login TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                email TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL
            )
        """)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)


def _is_auto_approved(email: str) -> bool:
    email = email.lower().strip()
    if email == ADMIN_EMAIL:
        return True
    # Use rsplit to safely get the domain part (handles edge cases)
    parts = email.rsplit("@", 1)
    domain = parts[1] if len(parts) == 2 else ""
    return domain in APPROVED_DOMAINS


def upsert_google_user(email: str, name: str, picture: str) -> dict:
    """
    Create or update a user from Google OAuth data.
    Returns the user dict with approval status.
    """
    email = email.lower().strip()
    conn = get_conn()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        now = datetime.utcnow().isoformat()

        c.execute("SELECT * FROM users WHERE email=%s", (email,))
        existing = c.fetchone()
        is_admin = 1 if email == ADMIN_EMAIL else 0
        auto_approved = _is_auto_approved(email)

        if existing:
            # Update profile info and last login
            c.execute("""
                UPDATE users SET name=%s, picture=%s, last_login=%s, is_admin=%s
                WHERE email=%s
            """, (name, picture, now, is_admin, email))
            # Auto-approve if domain matches but wasn't approved before
            if auto_approved and not existing["is_approved"]:
                c.execute("UPDATE users SET is_approved=1, approved_at=%s WHERE email=%s", (now, email))
            conn.commit()
            c.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = dict(c.fetchone())
        else:
            approved = 1 if auto_approved else 0
            c.execute("""
                INSERT INTO users (email, name, picture, is_approved, is_admin, approved_at, last_login)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (email, name, picture, approved, is_admin, now if approved else None, now))
            conn.commit()
            c.execute("SELECT * FROM users WHERE email=%s", (email,))
            user = dict(c.fetchone())

        return user
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)


def create_session(user: dict) -> str:
    """Create a session token for an approved user. Returns the token."""
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)).isoformat()
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO sessions (token, user_id, email, is_admin, expires_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (token, user["id"], user["email"], user["is_admin"], expires))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)
    return token


def validate_session(token: str):
    """Return session info if valid, None if expired/invalid."""
    if not token:
        return None
    conn = get_conn()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT * FROM sessions WHERE token=%s", (token,))
        session = c.fetchone()
        if not session:
            return None
        if datetime.utcnow().isoformat() > session["expires_at"]:
            return None
        return dict(session)
    finally:
        return_conn(conn)


def refresh_session(token: str):
    """Push the expiry out by SESSION_TTL_HOURS from now (sliding window)."""
    new_expiry = (datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)).isoformat()
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("UPDATE sessions SET expires_at=%s WHERE token=%s", (new_expiry, token))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)


def logout_user(token: str):
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("DELETE FROM sessions WHERE token=%s", (token,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)


def get_all_users() -> list:
    conn = get_conn()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute(
            "SELECT id, email, name, is_approved, is_admin, created_at, approved_at, last_login FROM users ORDER BY created_at DESC"
        )
        return [dict(r) for r in c.fetchall()]
    finally:
        return_conn(conn)


def get_pending_users() -> list:
    conn = get_conn()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute(
            "SELECT id, email, name, picture, created_at FROM users WHERE is_approved=0 ORDER BY created_at DESC"
        )
        return [dict(r) for r in c.fetchall()]
    finally:
        return_conn(conn)


def approve_user(user_id: int):
    conn = get_conn()
    try:
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute("UPDATE users SET is_approved=1, approved_at=%s WHERE id=%s", (now, user_id))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)


def revoke_user(user_id: int):
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("UPDATE users SET is_approved=0 WHERE id=%s", (user_id,))
        # Kill their active sessions
        c.execute("DELETE FROM sessions WHERE user_id=%s", (user_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)
