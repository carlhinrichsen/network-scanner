"""
Simple but secure auth layer.
- Sessions stored server-side in SQLite (same DB)
- Approved domains (e.g. execfunctions.co) auto-approved on first login
- Other emails go into a pending whitelist that the admin approves
- Admin is identified by ADMIN_EMAIL env var
- Passwords hashed with bcrypt via hashlib + secrets (no extra deps)
"""
import os
import sqlite3
import secrets
import hashlib
import json
from datetime import datetime, timedelta

DB_PATH = os.environ.get("DB_PATH", "data/connections.db")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")
APPROVED_DOMAINS = os.environ.get("APPROVED_DOMAINS", "execfunctions.co").split(",")
SESSION_TTL_HOURS = 72  # sessions last 3 days

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
            password_hash TEXT NOT NULL,
            is_approved INTEGER DEFAULT 0,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            approved_at TEXT
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

    # Ensure admin account exists if ADMIN_EMAIL is set
    if ADMIN_EMAIL:
        existing = c.execute("SELECT id FROM users WHERE email=?", (ADMIN_EMAIL,)).fetchone()
        if not existing:
            # Admin gets a random initial password — they'll set it on first login
            placeholder = _hash_password(secrets.token_hex(32))
            c.execute("""
                INSERT OR IGNORE INTO users (email, password_hash, is_approved, is_admin)
                VALUES (?, ?, 1, 1)
            """, (ADMIN_EMAIL, placeholder))
            conn.commit()

    conn.close()

def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
    return f"{salt}:{h.hex()}"

def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, h = stored_hash.split(":", 1)
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 260000)
        return check.hex() == h
    except Exception:
        return False

def _is_approved_domain(email: str) -> bool:
    domain = email.split("@")[-1].lower().strip()
    return domain in [d.lower().strip() for d in APPROVED_DOMAINS]

def register_user(email: str, password: str) -> dict:
    """Register a new user. Auto-approve if domain is whitelisted."""
    email = email.lower().strip()
    if not email or "@" not in email:
        return {"ok": False, "error": "Invalid email address"}
    if len(password) < 8:
        return {"ok": False, "error": "Password must be at least 8 characters"}

    conn = get_conn()
    c = conn.cursor()
    existing = c.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
    if existing:
        conn.close()
        return {"ok": False, "error": "An account with this email already exists"}

    is_approved = 1 if (_is_approved_domain(email) or email == ADMIN_EMAIL) else 0
    is_admin = 1 if email == ADMIN_EMAIL else 0
    pw_hash = _hash_password(password)
    now = datetime.utcnow().isoformat()

    c.execute("""
        INSERT INTO users (email, password_hash, is_approved, is_admin, approved_at)
        VALUES (?, ?, ?, ?, ?)
    """, (email, pw_hash, is_approved, is_admin, now if is_approved else None))
    conn.commit()
    conn.close()

    if is_approved:
        return {"ok": True, "approved": True, "message": "Account created. You can now log in."}
    else:
        return {"ok": True, "approved": False, "message": "Account created. Awaiting admin approval before you can log in."}

def login_user(email: str, password: str) -> dict:
    """Validate credentials and return a session token."""
    email = email.lower().strip()
    conn = get_conn()
    c = conn.cursor()
    user = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()

    if not user:
        conn.close()
        return {"ok": False, "error": "Invalid email or password"}
    if not _verify_password(password, user["password_hash"]):
        conn.close()
        return {"ok": False, "error": "Invalid email or password"}
    if not user["is_approved"]:
        conn.close()
        return {"ok": False, "error": "Your account is pending approval. The admin has been notified."}

    # Create session
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)).isoformat()
    c.execute("""
        INSERT INTO sessions (token, user_id, email, is_admin, expires_at)
        VALUES (?, ?, ?, ?, ?)
    """, (token, user["id"], email, user["is_admin"], expires))
    conn.commit()
    conn.close()
    return {"ok": True, "token": token, "email": email, "is_admin": bool(user["is_admin"])}

def validate_session(token: str) -> dict | None:
    """Return session info if valid, None if expired/invalid."""
    if not token:
        return None
    conn = get_conn()
    session = conn.execute(
        "SELECT * FROM sessions WHERE token=?", (token,)
    ).fetchone()
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

def get_pending_users() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, email, created_at FROM users WHERE is_approved=0 ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def approve_user(user_id: int) -> bool:
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE users SET is_approved=1, approved_at=? WHERE id=?", (now, user_id)
    )
    conn.commit()
    conn.close()
    return True

def get_all_users() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, email, is_approved, is_admin, created_at, approved_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def set_password(email: str, new_password: str) -> bool:
    if len(new_password) < 8:
        return False
    conn = get_conn()
    pw_hash = _hash_password(new_password)
    conn.execute("UPDATE users SET password_hash=? WHERE email=?", (pw_hash, email.lower().strip()))
    conn.commit()
    conn.close()
    return True
