import os
import io
import csv
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import init_db, upsert_connections, get_all_connections, get_stats, save_enrichment, log_upload
from csv_parser import parse_linkedin_csv
from ai_engine import (
    extract_filters, filter_connections, synthesise_response,
    compare_icps, enrich_company, chat_response
)

app = FastAPI(title="Network Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files if directory exists
static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup():
    init_db()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    message: str
    history: list = []
    mode: str = "search"  # "search" | "icp" | "chat"

class EnrichRequest(BaseModel):
    linkedin_urls: list[str]

class ICPRequest(BaseModel):
    description: str

# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    content = await file.read()
    try:
        rows = parse_linkedin_csv(content)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse CSV: {str(e)}")

    result = upsert_connections(rows)
    log_upload(file.filename, len(rows), result["added"], result["updated"], result["skipped"])

    return {
        "success": True,
        "filename": file.filename,
        "total_parsed": len(rows),
        **result
    }

# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

@app.get("/api/stats")
async def stats():
    return get_stats()

# ---------------------------------------------------------------------------
# Chat / Search endpoint
# ---------------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatMessage):
    connections = get_all_connections()
    stats = get_stats()
    db_context = f"{stats['total']} contacts, {stats['companies']} unique companies"

    if not connections:
        return {
            "type": "chat",
            "message": "No contacts loaded yet. Please upload your LinkedIn connections CSV first.",
            "results": [],
            "total": 0
        }

    # Detect ICP comparison intent
    icp_keywords = ["icp", "ideal customer", "compare", "persona", "profile", "which type", "target audience"]
    is_icp = req.mode == "icp" or any(kw in req.message.lower() for kw in icp_keywords)

    if is_icp and any(str(i) in req.message for i in range(2, 6)):
        # Multi-ICP comparison
        icp_results = compare_icps(connections, req.message)
        return {
            "type": "icp_comparison",
            "message": f"Quick estimate complete. Compared {len(icp_results)} ICPs against {len(connections)} contacts.",
            "icp_results": icp_results,
            "total": len(icp_results)
        }

    # Check if it's a search request or general chat
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
        except Exception as e:
            # Fall back to general chat if filter extraction fails
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
# Enrichment endpoint
# ---------------------------------------------------------------------------

@app.post("/api/enrich")
async def enrich(req: EnrichRequest):
    connections = get_all_connections()
    conn_map = {c["linkedin_url"]: c for c in connections}

    # Get unique companies for requested URLs
    to_enrich = {}
    for url in req.linkedin_urls:
        c = conn_map.get(url)
        if c and c.get("company") and not c.get("enriched_at"):
            company = c["company"]
            if company not in to_enrich:
                to_enrich[company] = []
            to_enrich[company].append(url)

    if not to_enrich:
        return {"message": "All selected contacts are already enriched or have no company.", "enriched": 0}

    enriched_count = 0
    results = []
    for company, urls in to_enrich.items():
        data = enrich_company(company)
        for url in urls:
            save_enrichment(url, data.get("industry", ""), data.get("description", ""), data.get("location", ""))
            enriched_count += 1
        results.append({
            "company": company,
            "industry": data.get("industry", ""),
            "description": data.get("description", ""),
            "location": data.get("location", ""),
            "contacts_updated": len(urls)
        })

    return {
        "message": f"Enriched {len(to_enrich)} companies ({enriched_count} contacts updated).",
        "enriched": enriched_count,
        "details": results
    }

# ---------------------------------------------------------------------------
# Export endpoint
# ---------------------------------------------------------------------------

class ExportRequest(BaseModel):
    results: list  # full result list from the frontend

@app.post("/api/export")
async def export_csv(req: ExportRequest):
    """Export exactly the results the frontend is holding — no re-filtering."""
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
async def root():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        with open(html_path) as f:
            return f.read()
    return HTMLResponse("<h1>Network Scanner API running. Frontend not found.</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
