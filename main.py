import os
import io
import csv
import json
import secrets
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

from database import init_db, upsert_connections, get_all_connections, get_stats, save_enrichment, log_upload
from csv_parser import parse_linkedin_csv
from ai_engine import (
    extract_filters, filter_connections, synthesise_response,
    compare_icps, enrich_contact, chat_response, discovery_response
)
from auth import (
    init_auth_db, upsert_google_user, create_session, validate_session,
    refresh_session, logout_user, get_pending_users, approve_user, revoke_user, get_all_users
)

# ---------------------------------------------------------------------------
# Google OAuth config
# ---------------------------------------------------------------------------
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
APP_BASE_URL         = os.environ.get("APP_BASE_URL", "https://shrimpsandwich.onrender.com")
GOOGLE_REDIRECT_URI  = f"{APP_BASE_URL}/api/auth/google/callback"

GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO  = "https://www.googleapis.com/oauth2/v3/userinfo"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Network Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[APP_BASE_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup():
    import database as _db
    import auth as _auth
    # Validate required env vars — warn clearly in logs rather than crashing
    for var in ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "ANTHROPIC_API_KEY"]:
        if not os.environ.get(var):
            print(f"[startup] WARNING: {var} is not set — some features will not work")
    db_url = _db.DATABASE_URL
    masked = db_url[:30] + "..." if len(db_url) > 30 else db_url
    print(f"[startup] DATABASE_URL: {masked}")
    if not db_url:
        print("[startup] WARNING: DATABASE_URL is not set — database will not work")
    init_db()
    init_auth_db()
    # Log contact count so we can confirm data survived the deploy
    try:
        stats = get_stats()
        print(f"[startup] Contacts in DB: {stats['total']}, enriched: {stats['enriched']}")
    except Exception as e:
        print(f"[startup] Could not read stats: {e}")

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_session(request: Request):
    token = request.cookies.get("ns_token") or request.headers.get("X-Session-Token")
    session = validate_session(token)
    if session and token:
        refresh_session(token)   # sliding window — active users never time out
    return session

def require_session(request: Request) -> dict:
    session = get_session(request)
    if not session:
        raise HTTPException(401, "Authentication required")
    return session

def require_admin(request: Request) -> dict:
    session = require_session(request)
    if not session.get("is_admin"):
        raise HTTPException(403, "Admin access required")
    return session

# ---------------------------------------------------------------------------
# Google OAuth endpoints
# ---------------------------------------------------------------------------

@app.get("/api/auth/google/login")
async def google_login(response: Response):
    """Redirect the browser to Google's OAuth consent screen."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(500, "Google OAuth not configured. Set GOOGLE_CLIENT_ID env var.")

    state = secrets.token_urlsafe(16)
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "state":         state,
        "prompt":        "select_account",
    }
    from urllib.parse import urlencode
    url = f"{GOOGLE_AUTH_URL}?{urlencode(params)}"
    resp = RedirectResponse(url)
    # Store state in a short-lived cookie for CSRF protection
    resp.set_cookie("oauth_state", state, httponly=True, samesite="lax", max_age=600)
    return resp

@app.get("/api/auth/google/callback")
async def google_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Handle Google's redirect back after user authenticates."""
    if error:
        return RedirectResponse(f"/?auth_error={error}")

    # CSRF check
    stored_state = request.cookies.get("oauth_state", "")
    if stored_state and state and stored_state != state:
        return RedirectResponse("/?auth_error=state_mismatch")

    # Exchange code for tokens
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(GOOGLE_TOKEN_URL, data={
            "code":          code,
            "client_id":     GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri":  GOOGLE_REDIRECT_URI,
            "grant_type":    "authorization_code",
        })
        if token_resp.status_code != 200:
            return RedirectResponse("/?auth_error=token_exchange_failed")
        tokens = token_resp.json()

        # Fetch user info from Google
        userinfo_resp = await client.get(
            GOOGLE_USERINFO,
            headers={"Authorization": f"Bearer {tokens['access_token']}"}
        )
        if userinfo_resp.status_code != 200:
            return RedirectResponse("/?auth_error=userinfo_failed")
        userinfo = userinfo_resp.json()

    email   = userinfo.get("email", "")
    name    = userinfo.get("name", "")
    picture = userinfo.get("picture", "")

    if not email:
        return RedirectResponse("/?auth_error=no_email")

    # Upsert user in DB
    user = upsert_google_user(email, name, picture)

    if not user["is_approved"]:
        # Not approved yet — redirect with a pending message (no email in URL)
        return RedirectResponse("/?auth_pending=1")

    # Approved — create session and redirect home
    token = create_session(user)
    resp = RedirectResponse("/")
    resp.set_cookie(
        "ns_token", token,
        httponly=True, samesite="lax",
        max_age=720 * 3600,  # 30 days
        secure=APP_BASE_URL.startswith("https")
    )
    resp.delete_cookie("oauth_state")
    return resp

@app.post("/api/auth/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("ns_token")
    if token:
        logout_user(token)
    response.delete_cookie("ns_token")
    return {"ok": True}

@app.get("/api/auth/me")
async def me(request: Request):
    session = get_session(request)
    if not session:
        return {"authenticated": False}
    return {
        "authenticated": True,
        "email":    session["email"],
        "is_admin": bool(session["is_admin"])
    }

# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

@app.get("/api/admin/users")
async def admin_users(request: Request):
    require_admin(request)
    return get_all_users()

@app.get("/api/admin/pending")
async def admin_pending(request: Request):
    require_admin(request)
    return get_pending_users()

@app.post("/api/admin/approve/{user_id}")
async def admin_approve(user_id: int, request: Request):
    require_admin(request)
    approve_user(user_id)
    return {"ok": True}

@app.post("/api/admin/revoke/{user_id}")
async def admin_revoke(user_id: int, request: Request):
    require_admin(request)
    revoke_user(user_id)
    return {"ok": True}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    message: str
    history: list = []
    mode: str = "search"
    guest_data: list = []
    auto_enrich_threshold: int = 50          # ICP: auto-enrich if unenriched <= this
    proposed_searches: list = []             # Discovery: pre-approved web search queries
    proposed_db_search: str = ""             # Discovery: pre-approved DB search
    icp_confirmed_enrich: bool = False       # ICP: user confirmed large enrichment
    icp_descriptions: str = ""              # ICP: re-passed when confirming enrichment
    current_result_urls: list = []          # Frontend current result set (for location_blocked scoping)

class EnrichRequest(BaseModel):
    linkedin_urls: List[str]

class ExportRequest(BaseModel):
    results: list

# ---------------------------------------------------------------------------
# Shared CSV helper
# ---------------------------------------------------------------------------

async def _read_and_parse_csv(file: UploadFile) -> list:
    """Validate, read, and parse a LinkedIn CSV upload. Raises HTTPException on error."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    content = await file.read()
    try:
        return parse_linkedin_csv(content)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse CSV: {str(e)}")


# ---------------------------------------------------------------------------
# Upload — private (requires login)
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    require_session(request)
    rows = await _read_and_parse_csv(file)
    result = upsert_connections(rows)
    log_upload(file.filename, len(rows), result["added"], result["updated"], result["skipped"])
    return {"success": True, "filename": file.filename, "total_parsed": len(rows), **result}


# ---------------------------------------------------------------------------
# Guest upload — session only, never writes to DB
# ---------------------------------------------------------------------------

@app.post("/api/guest/upload")
async def guest_upload(file: UploadFile = File(...)):
    rows = await _read_and_parse_csv(file)
    return {"success": True, "total_parsed": len(rows), "contacts": rows}

# ---------------------------------------------------------------------------
# Stats — private
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def stats(request: Request):
    require_session(request)
    return get_stats()

# ---------------------------------------------------------------------------
# Chat / Search — private + guest
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatMessage, request: Request):
    session = get_session(request)
    is_private = bool(session)

    if is_private:
        connections = get_all_connections()
        db_context = f"{len(connections)} stored contacts"
    elif req.guest_data:
        connections = req.guest_data
        db_context = f"{len(connections)} uploaded contacts (guest session)"
    else:
        return {
            "type": "chat",
            "message": "Please log in to access stored contacts, or upload a CSV to use guest mode.",
            "results": [], "total": 0
        }

    if not connections:
        return {
            "type": "chat",
            "message": "No contacts loaded yet. Please upload your LinkedIn connections CSV first.",
            "results": [], "total": 0
        }

    # ── Discovery mode ──────────────────────────────────────────────────────
    if req.mode == "discovery":
        if not is_private:
            return {
                "type": "chat",
                "message": "Discovery mode requires a private account. Please log in to use web research and deep analysis.",
                "results": [], "total": 0
            }
        db_stats = get_stats()
        result = discovery_response(
            user_message=req.message,
            conversation_history=req.history,
            connections=connections,
            db_stats=db_stats,
            proposed_searches=req.proposed_searches or [],
            proposed_db_search=req.proposed_db_search or None,
        )
        return result

    # ── ICP mode ────────────────────────────────────────────────────────────
    icp_keywords = ["icp", "ideal customer", "compare", "persona", "profile", "which type", "target audience"]
    is_icp = req.mode == "icp" or any(kw in req.message.lower() for kw in icp_keywords)
    if is_icp:
        # When user confirmed a large enrichment, re-run with full enrich
        icp_desc = req.icp_descriptions or req.message
        threshold = req.auto_enrich_threshold if not req.icp_confirmed_enrich else len(connections)

        enrich_fn = enrich_contact if is_private else None
        save_fn = save_enrichment if is_private else None

        icp_data = compare_icps(
            connections=connections,
            icp_descriptions=icp_desc,
            auto_enrich_threshold=threshold,
            enrich_fn=enrich_fn,
            save_fn=save_fn,
        )

        if icp_data["enrichment_needed"]:
            # Return a special type so frontend shows confirmation UI
            unenriched = icp_data["unenriched_count"]
            return {
                "type": "icp_enrich_needed",
                "message": f"🔍 I found **{unenriched} unenriched contacts** in your network. Enriching them will make ICP scoring much more accurate (location, industry, company descriptions).\n\nThis will take ~{max(1, unenriched // 10)} minutes. Proceed?",
                "unenriched_count": unenriched,
                "icp_descriptions": icp_desc,
                "results": [], "total": 0
            }

        icp_results = icp_data["results"]
        enrichment_note = " ✨ Contacts were enriched before scoring." if icp_data["enrichment_performed"] else ""
        return {
            "type": "icp_comparison",
            "message": f"Compared {len(icp_results)} ICPs against {len(connections)} contacts.{enrichment_note}",
            "icp_results": icp_results,
            "total": len(icp_results)
        }

    # ── Search mode ─────────────────────────────────────────────────────────
    search_keywords = [
        "find", "show", "search", "look for", "who", "list", "give me",
        "filter", "identify", "get me", "fetch", "people", "contacts",
        "anyone", "somebody", "heads of", "vp", "director", "ceo", "cto",
        "manager", "founder", "engineer", "sales", "marketing", "product"
    ]
    is_search = req.mode == "search" or any(kw in req.message.lower() for kw in search_keywords)

    if is_search:
        try:
            filters = extract_filters(req.message, req.history)
        except Exception:
            msg = chat_response(req.message, req.history, db_context)
            return {"type": "chat", "message": msg, "results": [], "total": 0}

        # Location queries require enrichment data — guide user if data is missing
        if filters.get("requires_location"):
            # Scope check to the current result set (sent by frontend), not the whole DB
            pool = connections
            if req.current_result_urls:
                url_set = set(req.current_result_urls)
                pool = [c for c in connections if c.get("linkedin_url") in url_set]
            has_any_location = any(c.get("location") for c in pool)
            if not has_any_location:
                return {
                    "type": "location_blocked",
                    "message": (
                        "📍 **None of the current contacts have location data yet.**\n\n"
                        "Select contacts and click **✨ Enrich** to fetch their locations, "
                        "then refine by location."
                    ),
                }
            # else: at least one has location — allow the search through

        results = filter_connections(connections, filters)
        summary = synthesise_response(req.message, filters, results, req.history)
        return {
            "type": "search",
            "message": summary,
            "intent": filters.get("intent_summary", ""),
            "results": results,
            "total": len(results),
            "preview_count": min(20, len(results)),
            "filters": filters
        }
    else:
        msg = chat_response(req.message, req.history, db_context)
        return {"type": "chat", "message": msg, "results": [], "total": 0}

# ---------------------------------------------------------------------------
# Enrichment — private only
# ---------------------------------------------------------------------------

@app.post("/api/enrich")
async def enrich(req: EnrichRequest, request: Request):
    require_session(request)
    connections = get_all_connections()
    conn_map = {c["linkedin_url"]: c for c in connections}

    to_enrich = {}
    for url in req.linkedin_urls:
        c = conn_map.get(url)
        if c and not c.get("enriched_at"):
            company = c.get("company") or "Unknown"
            if company not in to_enrich:
                to_enrich[company] = []
            to_enrich[company].append(url)

    if not to_enrich:
        return {"message": "All selected contacts are already enriched.", "enriched": 0}

    enriched_count = 0
    results = []
    company_cache: dict = {}

    for company, urls in to_enrich.items():
        url_locations: dict = {}
        for url in urls:
            data = enrich_contact(url, company)
            location    = data.get("location", "")
            industry    = data.get("industry", "")
            description = data.get("description", "")

            if (not industry or not description) and company in company_cache:
                industry    = industry    or company_cache[company].get("industry", "")
                description = description or company_cache[company].get("description", "")

            if company not in company_cache:
                company_cache[company] = {"industry": industry, "description": description}

            url_locations[url] = location
            save_enrichment(url, industry, description, location)
            enriched_count += 1

        results.append({
            "company":          company,
            "industry":         company_cache.get(company, {}).get("industry", ""),
            "description":      company_cache.get(company, {}).get("description", ""),
            "contacts_updated": len(urls),
            "url_locations":    url_locations,
        })

    return {
        "message":  f"Enriched {enriched_count} contacts across {len(to_enrich)} companies.",
        "enriched": enriched_count,
        "details":  results
    }

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/healthz")
async def healthz():
    try:
        s = get_stats()
        return {"status": "ok", "contacts": s["total"], "enriched": s["enriched"]}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# Export — works for both modes
# ---------------------------------------------------------------------------

@app.post("/api/export")
async def export_csv(req: ExportRequest):
    output = io.StringIO()
    fieldnames = [
        "first_name", "last_name", "company", "position",
        "location", "linkedin_url", "connected_on",
        "enriched_industry", "enriched_company_desc", "_score"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(req.results)
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=network_scan_export.csv"}
    )

# ---------------------------------------------------------------------------
# Serve frontend (catch-all)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/{path:path}", response_class=HTMLResponse)
async def root(path: str = ""):
    # Don't catch API or static routes
    if path.startswith("api/") or path.startswith("static/"):
        raise HTTPException(404)
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return HTMLResponse("<h1>Network Scanner API running.</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
