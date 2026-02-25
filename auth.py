"""
Auth layer — Google OAuth only.
- No passwords stored anywhere
- Google verifies the email actually exists and belongs to the user
- Auto-approve logic: execfunctions.co domain (or any APPROVED_DOMAINS) + admin whitelist
- Sessions stored server-side in SQLite
"""
import os
import secrets
from datetime import datetime, timedelta
import sqlite3

DB_PATH = os.environ.get("DB_PATH", "data/connections.db")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "").lower().strip()
APPROVED_DOMAINS = [d.lower().strip() for d in os.environ.get("APPROVED_DOMAINS", "execfunctions.co").split(",") if d.strip()]
SESSION_TTL_HOURS = 72

def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_auth_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    conn.close()

def _is_auto_approved(email: str) -> bool:
    email = email.lower().strip()
    if email == ADMIN_EMAIL:
        return True
    domain = email.split("@")[-1]
    return domain in APPROVED_DOMAINS

def upsert_google_user(email: str, name: str, picture: str) -> dict:
    """
    Create or update a user from Google OAuth data.
    Returns the user dict with approval status.
    """
    email = email.lower().strip()
    conn = get_conn()
    c = conn.cursor()
    now = datetime.utcnow().isoformat()

    existing = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    is_admin = 1 if email == ADMIN_EMAIL else 0
    auto_approved = _is_auto_approved(email)

    if existing:
        # Update profile info and last login
        c.execute("""
            UPDATE users SET name=?, picture=?, last_login=?, is_admin=?
            WHERE email=?
        """, (name, picture, now, is_admin, email))
        # Auto-approve if domain matches but wasn't approved before
        if auto_approved and not existing["is_approved"]:
            c.execute("UPDATE users SET is_approved=1, approved_at=? WHERE email=?", (now, email))
        conn.commit()
        user = dict(c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone())
    else:
        approved = 1 if auto_approved else 0
        c.execute("""
            INSERT INTO users (email, name, picture, is_approved, is_admin, approved_at, last_login)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (email, name, picture, approved, is_admin, now if approved else None, now))
        conn.commit()
        user = dict(c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone())

    conn.close()
    return user

def create_session(user: dict) -> str:
    """Create a session token for an approved user. Returns the token."""
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)).isoformat()
    conn = get_conn()
    conn.execute("""
        INSERT INTO sessions (token, user_id, email, is_admin, expires_at)
        VALUES (?, ?, ?, ?, ?)
    """, (token, user["id"], user["email"], user["is_admin"], expires))
    conn.commit()
    conn.close()
    return token

def validate_session(token: str) -> dict | None:
    """Return session info if valid, None if expired/invalid."""
    if not token:
        return None
    conn = get_conn()
    session = conn.execute("SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
    conn.close()
    if not session:
        return None
    if datetime.utcnow().isoformat() > session["expires_at"]:
        return None
    return dict(session)

def logout_user(token: str):
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()

def get_all_users() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, email, name, is_approved, is_admin, created_at, approved_at, last_login FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_pending_users() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, email, name, picture, created_at FROM users WHERE is_approved=0 ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def approve_user(user_id: int):
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute("UPDATE users SET is_approved=1, approved_at=? WHERE id=?", (now, user_id))
    conn.commit()
    conn.close()

def revoke_user(user_id: int):
    conn = get_conn()
    conn.execute("UPDATE users SET is_approved=0 WHERE id=?", (user_id,))
    # Kill their active sessions
    conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
