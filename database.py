import os
import psycopg2
import psycopg2.extras
import psycopg2.pool
from datetime import datetime
from typing import List

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# ThreadedConnectionPool is thread-safe — required because uvicorn runs
# FastAPI sync handlers on threads. Lazily initialized so import doesn't
# fail if DATABASE_URL isn't set yet (e.g. during local dev without env).
_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            dsn=DATABASE_URL,
        )
    return _pool


def get_conn():
    conn = _get_pool().getconn()
    conn.autocommit = False
    return conn


def return_conn(conn):
    _get_pool().putconn(conn)


def init_db():
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS connections (
                id SERIAL PRIMARY KEY,
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
        c.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'connections' AND table_schema = 'public'
        """)
        existing_cols = [row[0] for row in c.fetchall()]
        if "location" not in existing_cols:
            c.execute("ALTER TABLE connections ADD COLUMN location TEXT")
        c.execute("""
            CREATE TABLE IF NOT EXISTS upload_log (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                total_rows INTEGER,
                added INTEGER,
                updated INTEGER,
                skipped INTEGER,
                uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)


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
            c.execute(
                "SELECT id FROM connections WHERE linkedin_url = %s", (url,)
            )
            existing = c.fetchone()
            if existing:
                c.execute("""
                    UPDATE connections SET
                        first_name=%s, last_name=%s, email=%s, company=%s,
                        position=%s, connected_on=%s, updated_at=%s
                    WHERE linkedin_url=%s
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
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    url, row.get("first_name"), row.get("last_name"),
                    row.get("email"), row.get("company"),
                    row.get("position"), row.get("connected_on"),
                    now, now
                ))
                added += 1

        conn.commit()
        return {"added": added, "updated": updated, "skipped": skipped}
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)


def get_all_connections() -> List[dict]:
    conn = get_conn()
    try:
        c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        c.execute("SELECT * FROM connections ORDER BY connected_on DESC")
        return [dict(r) for r in c.fetchall()]
    finally:
        return_conn(conn)


def get_stats() -> dict:
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM connections")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM connections WHERE enriched_at IS NOT NULL")
        enriched = c.fetchone()[0]
        c.execute(
            "SELECT COUNT(DISTINCT company) FROM connections WHERE company IS NOT NULL AND company != ''"
        )
        companies = c.fetchone()[0]
        c.execute(
            "SELECT uploaded_at, filename FROM upload_log ORDER BY id DESC LIMIT 1"
        )
        last_upload = c.fetchone()
        return {
            "total": total,
            "enriched": enriched,
            "companies": companies,
            "last_upload": {"uploaded_at": last_upload[0], "filename": last_upload[1]} if last_upload else None
        }
    finally:
        return_conn(conn)


def save_enrichment(linkedin_url: str, industry: str, description: str, location: str = ""):
    conn = get_conn()
    try:
        c = conn.cursor()
        now = datetime.utcnow().isoformat()
        c.execute("""
            UPDATE connections SET
                enriched_industry=%s, enriched_company_desc=%s, location=%s, enriched_at=%s
            WHERE linkedin_url=%s
        """, (industry, description, location, now, linkedin_url))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)


def log_upload(filename: str, total: int, added: int, updated: int, skipped: int):
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("""
            INSERT INTO upload_log (filename, total_rows, added, updated, skipped)
            VALUES (%s, %s, %s, %s, %s)
        """, (filename, total, added, updated, skipped))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        return_conn(conn)
