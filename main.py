import os
import io
import csv
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import init_db, upsert_connections, get_all_connections, get_stats, save_enrichment, log_upload
from csv_parser import parse_linkedin_csv
from ai_engine import (
    extract_filters, filter_connections, synthesise_response,
    compare_icps, enrich_contact, chat_response
)
from auth import (
    init_auth_db, register_user, login_user, validate_session,
    logout_user, get_pending_users, approve_user, get_all_users, set_password
)

app = FastAPI(title="Network Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup():
    init_db()
    init_auth_db()

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def get_session(request: Request) -> dict | None:
    token = request.cookies.get("ns_token") or request.headers.get("X-Session-Token")
    return validate_session(token)

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
# Auth endpoints
# ---------------------------------------------------------------------------

class AuthRequest(BaseModel):
    email: str
    password: str

@app.post("/api/auth/register")
async def register(req: AuthRequest):
    return register_user(req.email, req.password)

@app.post("/api/auth/login")
async def login(req: AuthRequest, response: Response):
    result = login_user(req.email, req.password)
    if result.get("ok"):
        response.set_cookie(
            "ns_token", result["token"],
            httponly=True, samesite="lax",
            max_age=72 * 3600
        )
    return result

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
    return {"authenticated": True, "email": session["email"], "is_admin": bool(session["is_admin"])}

@app.post("/api/auth/set-password")
async def change_password(req: AuthRequest, request: Request):
    session = require_session(request)
    if session["email"] != req.email.lower().strip() and not session["is_admin"]:
        raise HTTPException(403, "Cannot change another user's password")
    ok = set_password(req.email, req.password)
    return {"ok": ok}

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

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    message: str
    history: list = []
    mode: str = "search"
    guest_data: list = []  # guest mode passes their session data directly

class EnrichRequest(BaseModel):
    linkedin_urls: list[str]

class ExportRequest(BaseModel):
    results: list

# ---------------------------------------------------------------------------
# Upload endpoint — PRIVATE (requires login)
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    require_session(request)
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    content = await file.read()
    try:
        rows = parse_linkedin_csv(content)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse CSV: {str(e)}")

    result = upsert_connections(rows)
    log_upload(file.filename, len(rows), result["added"], result["updated"], result["skipped"])
    return {"success": True, "filename": file.filename, "total_parsed": len(rows), **result}

# ---------------------------------------------------------------------------
# Guest upload endpoint — session only, returns data, does NOT write to DB
# ---------------------------------------------------------------------------

@app.post("/api/guest/upload")
async def guest_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    content = await file.read()
    try:
        rows = parse_linkedin_csv(content)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse CSV: {str(e)}")
    # Return the parsed rows directly — client holds them in memory
    return {"success": True, "total_parsed": len(rows), "contacts": rows}

# ---------------------------------------------------------------------------
# Stats endpoint — PRIVATE
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def stats(request: Request):
    require_session(request)
    return get_stats()

# ---------------------------------------------------------------------------
# Chat / Search — works in both private and guest mode
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatMessage, request: Request):
    session = get_session(request)

    # Determine data source
    if session:
        # Private mode: use persistent DB
        connections = get_all_connections()
        db_context = f"{len(connections)} stored contacts"
    elif req.guest_data:
        # Guest mode: use data passed from client
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

    # ICP comparison
    icp_keywords = ["icp", "ideal customer", "compare", "persona", "profile", "which type", "target audience"]
    is_icp = req.mode == "icp" or any(kw in req.message.lower() for kw in icp_keywords)
    if is_icp and any(str(i) in req.message for i in range(2, 6)):
        icp_results = compare_icps(connections, req.message)
        return {
            "type": "icp_comparison",
            "message": f"Quick estimate complete. Compared {len(icp_results)} ICPs against {len(connections)} contacts.",
            "icp_results": icp_results,
            "total": len(icp_results)
        }

    # Search or chat
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
# Enrichment — PRIVATE only
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
        for url in urls:
            data = enrich_contact(url, company)
            location = data.get("location", "")
            industry = data.get("industry", "")
            description = data.get("description", "")

            if (not industry or not description) and company in company_cache:
                industry = industry or company_cache[company].get("industry", "")
                description = description or company_cache[company].get("description", "")

            if company not in company_cache:
                company_cache[company] = {"industry": industry, "description": description}

            save_enrichment(url, industry, description, location)
            enriched_count += 1

        results.append({
            "company": company,
            "industry": company_cache.get(company, {}).get("industry", ""),
            "description": company_cache.get(company, {}).get("description", ""),
            "contacts_updated": len(urls)
        })

    return {
        "message": f"Enriched {enriched_count} contacts across {len(to_enrich)} companies.",
        "enriched": enriched_count,
        "details": results
    }

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
# Serve frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/{path:path}", response_class=HTMLResponse)
async def root(path: str = ""):
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return HTMLResponse("<h1>Network Scanner API running.</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
