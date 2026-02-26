import sqlite3
import os
from datetime import datetime
from typing import List

DB_PATH = os.environ.get("DB_PATH", "data/connections.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                linkedin_url TEXT UNIQUE NOT NULL,
                first_name TEXT,
                last_name TEXT,
                email TEXT,
                company TEXT,
                position TEXT,
                connected_on TEXT,
                location TEXT,
                enriched_industry TEXT,
                enriched_company_desc TEXT,
                enriched_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Migrate existing databases that don't have the location column yet
        existing_cols = [row[1] for row in c.execute("PRAGMA table_info(connections)").fetchall()]
        if "location" not in existing_cols:
            c.execute("ALTER TABLE connections ADD COLUMN location TEXT")
        c.execute("""
            CREATE TABLE IF NOT EXISTS upload_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                total_rows INTEGER,
                added INTEGER,
                updated INTEGER,
                skipped INTEGER,
                uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()


def upsert_connections(rows: List[dict]) -> dict:
    conn = get_conn()
    try:
        c = conn.cursor()
        added = updated = skipped = 0
        now = datetime.utcnow().isoformat()

        for row in rows:
            url = row.get("linkedin_url", "").strip()
            if not url:
                skipped += 1
                continue
            existing = c.execute(
                "SELECT id FROM connections WHERE linkedin_url = ?", (url,)
            ).fetchone()
            if existing:
                c.execute("""
                    UPDATE connections SET
                        first_name=?, last_name=?, email=?, company=?,
                        position=?, connected_on=?, updated_at=?
                    WHERE linkedin_url=?
                """, (
                    row.get("first_name"), row.get("last_name"),
                    row.get("email"), row.get("company"),
                    row.get("position"), row.get("connected_on"),
                    now, url
                ))
                updated += 1
            else:
                c.execute("""
                    INSERT INTO connections
                        (linkedin_url, first_name, last_name, email,
                         company, position, connected_on, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    url, row.get("first_name"), row.get("last_name"),
                    row.get("email"), row.get("company"),
                    row.get("position"), row.get("connected_on"),
                    now, now
                ))
                added += 1

        conn.commit()
        return {"added": added, "updated": updated, "skipped": skipped}
    finally:
        conn.close()


def get_all_connections() -> List[dict]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM connections ORDER BY connected_on DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_stats() -> dict:
    conn = get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM connections").fetchone()[0]
        enriched = conn.execute(
            "SELECT COUNT(*) FROM connections WHERE enriched_at IS NOT NULL"
        ).fetchone()[0]
        companies = conn.execute(
            "SELECT COUNT(DISTINCT company) FROM connections WHERE company IS NOT NULL AND company != ''"
        ).fetchone()[0]
        last_upload = conn.execute(
            "SELECT uploaded_at, filename FROM upload_log ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return {
            "total": total,
            "enriched": enriched,
            "companies": companies,
            "last_upload": dict(last_upload) if last_upload else None
        }
    finally:
        conn.close()


def save_enrichment(linkedin_url: str, industry: str, description: str, location: str = ""):
    conn = get_conn()
    try:
        now = datetime.utcnow().isoformat()
        conn.execute("""
            UPDATE connections SET
                enriched_industry=?, enriched_company_desc=?, location=?, enriched_at=?
            WHERE linkedin_url=?
        """, (industry, description, location, now, linkedin_url))
        conn.commit()
    finally:
        conn.close()


def log_upload(filename: str, total: int, added: int, updated: int, skipped: int):
    conn = get_conn()
    try:
        conn.execute("""
            INSERT INTO upload_log (filename, total_rows, added, updated, skipped)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, total, added, updated, skipped))
        conn.commit()
    finally:
        conn.close()
