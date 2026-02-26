import os
import re
import json
import anthropic
from tavily import TavilyClient

# ---------------------------------------------------------------------------
# CONFIG CONSTANTS — change models / thresholds here, not inline
# ---------------------------------------------------------------------------

HAIKU_MODEL  = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-4-5"

ICP_MIN_THRESHOLD  = 0.60   # floor for ICP confidence_threshold
ICP_THRESHOLD      = 0.65   # default ICP confidence_threshold
ICP_HIGH_THRESHOLD = 0.75   # "high confidence" ICP match
SEARCH_THRESHOLD   = 0.50   # default search confidence_threshold

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TAVILY_KEY = os.environ.get("TAVILY_KEY", "") or os.environ.get("TAVILY_API_KEY", "")


def get_claude():
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)


def get_tavily():
    return TavilyClient(api_key=TAVILY_KEY)


def _parse_json_response(raw: str) -> dict | list:
    """
    Safely parse JSON from a Claude response that may be wrapped in markdown fences.
    Returns {} (or []) on any parse error rather than raising.
    """
    text = raw.strip()
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# SEMANTIC FILTER
# ---------------------------------------------------------------------------

FILTER_SYSTEM = """You are a B2B network intelligence assistant helping analyse a LinkedIn connections database.

Your job: given a natural language search request, extract structured filter criteria and return ONLY valid JSON.

The database has these fields:
- first_name, last_name (the contact's name)
- company (column E in LinkedIn export)
- position (column F in LinkedIn export)
- connected_on (date connected on LinkedIn, format "DD MMM YYYY" e.g. "15 Jan 2023")
- location (city/country — populated only after enrichment)
- enriched_industry (may be empty — populated after enrichment)
- enriched_company_desc (may be empty — free-text company description, populated after enrichment)

Return JSON with this exact schema:
{
  "intent_summary": "1-2 sentence plain English summary of what the user is looking for",
  "search_mode": "new",
  "name_keywords": ["andrea", "smith"],
  "location_keywords": ["london", "uk", "berlin"],
  "requires_location": false,
  "company_keywords": ["list", "of", "company", "name", "fragments"],
  "position_keywords": ["list", "of", "job", "title", "keywords"],
  "position_concepts": ["broader role concepts to match against, e.g. 'revenue leader', 'technical founder'"],
  "seniority_levels": ["C-suite", "VP", "Director", "Manager", "Individual Contributor", "Founder"],
  "industries": ["list of industries if mentioned"],
  "description_keywords": ["b2b", "saas", "series a", "marketplace"],
  "connected_after_year": null,
  "connected_before_year": null,
  "exclude_keywords": ["words that should NOT appear in any field"],
  "confidence_threshold": 0.6
}

Field guidance:
- search_mode: CRITICAL — one of exactly "new", "narrow", or "expand":
  * "new" — the user wants a fresh search, unrelated to what's currently shown (e.g. "find me all founders", "show me people at Google", "search for VPs")
  * "narrow" — the user wants to filter DOWN the current results to a smaller subset (e.g. "now only the ones in SaaS", "just show me the UK ones", "remove anyone in sales", "filter to C-suite only"). Clue: words like "now only", "just the", "filter to", "narrow", "of those", "from those"
  * "expand" — the user wants to ADD more contacts to the existing list without losing current results (e.g. "also include marketing people", "add anyone from Stripe", "plus fintech contacts"). Clue: words like "also", "add", "plus", "include", "as well as", "and also"
  Use conversation history to understand whether there is an active result set being refined.
- name_keywords: use when user asks for contacts by first or last name (e.g. "all Andreas", "people called Smith"). Match fragments case-insensitively.
- location_keywords: city, country, or region fragments (e.g. "london", "uk", "germany", "nordics"). Set requires_location: true when the query is primarily location-based.
- requires_location: true ONLY if location is the primary filter (e.g. "who do I know in Paris"). False for queries that mention location incidentally.
- description_keywords: free-text concepts to find in enriched company descriptions (e.g. "b2b saas", "fintech", "series a", "marketplace", "enterprise"). Use when user asks about company type or stage.
- connected_after_year: integer year (e.g. 2023) — include only contacts connected on or after January 1st of this year. Use when user says "in 2025", "since 2024", "this year", "last year", etc. Set to null if no year constraint.
- connected_before_year: integer year — include only contacts connected before January 1st of this year. Set to null if no constraint.
- For "in 2025" → set connected_after_year: 2025 AND connected_before_year: 2026.
- For position_concepts, think broadly. E.g. "head of sales" should match: VP Sales, Chief Revenue Officer, Director of Sales, Head of Revenue, Sales Director, Commercial Director etc.
- For seniority_levels, only include if the user specifies seniority.
- All arrays can be empty [].
Return ONLY the JSON object, no markdown, no explanation."""


def extract_filters(user_message: str, conversation_history: list) -> dict:
    client = get_claude()
    messages = conversation_history[-6:] + [{"role": "user", "content": user_message}]

    resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=1024,
        system=FILTER_SYSTEM,
        messages=messages
    )
    result = _parse_json_response(resp.content[0].text)
    # Ensure result is a dict (not a list, not {})
    return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def score_connection(conn: dict, filters: dict) -> float:
    score = 0.0
    weights = 0.0
    position  = (conn.get("position") or "").lower()
    company   = (conn.get("company") or "").lower()
    industry  = (conn.get("enriched_industry") or "").lower()
    desc      = (conn.get("enriched_company_desc") or "").lower()
    location  = (conn.get("location") or "").lower()
    full_name = f"{conn.get('first_name') or ''} {conn.get('last_name') or ''}".lower().strip()

    # Name keyword match (first_name / last_name)
    nk = filters.get("name_keywords", [])
    if nk:
        weights += 3.0
        if any(kw.lower() in full_name for kw in nk):
            score += 3.0

    # Company keyword match
    ck = filters.get("company_keywords", [])
    if ck:
        weights += 3.0
        for kw in ck:
            if kw.lower() in company:
                score += 3.0
                break

    # Position keyword match
    pk = filters.get("position_keywords", [])
    if pk:
        weights += 2.0
        matched = sum(1 for kw in pk if kw.lower() in position)
        score += 2.0 * min(matched / max(len(pk), 1), 1.0)

    # Position concept match (broader)
    pc = filters.get("position_concepts", [])
    if pc:
        weights += 2.0
        for concept in pc:
            words = concept.lower().split()
            if any(w in position for w in words):
                score += 2.0
                break

    # Seniority match
    sl = filters.get("seniority_levels", [])
    if sl:
        weights += 1.5
        seniority_map = {
            "C-suite": ["ceo", "cto", "coo", "cfo", "cmo", "cro", "chief"],
            "VP": ["vp", "vice president", "vice-president"],
            "Director": ["director"],
            "Manager": ["manager", "lead", "head of"],
            "Individual Contributor": ["engineer", "designer", "analyst", "associate", "specialist"],
            "Founder": ["founder", "co-founder", "cofounder", "owner", "principal"],
        }
        for level in sl:
            keywords = seniority_map.get(level, [level.lower()])
            if any(kw in position for kw in keywords):
                score += 1.5
                break

    # Industry match (enriched data)
    industries = filters.get("industries", [])
    if industries:
        weights += 1.0
        for ind in industries:
            if ind.lower() in industry or ind.lower() in desc:
                score += 1.0
                break

    # Location keyword match (enriched field — may be empty for unenriched contacts)
    lk = filters.get("location_keywords", [])
    if lk:
        weights += 2.0
        if any(kw.lower() in location for kw in lk):
            score += 2.0

    # Description keyword match (free-text enriched_company_desc)
    dk = filters.get("description_keywords", [])
    if dk:
        weights += 1.0
        if any(kw.lower() in desc for kw in dk):
            score += 1.0

    # Connection date filter — hard gate (not a soft score)
    # LinkedIn connected_on format: "DD MMM YYYY" e.g. "15 Jan 2023"
    after_year  = filters.get("connected_after_year")
    before_year = filters.get("connected_before_year")
    if after_year is not None or before_year is not None:
        raw_date = conn.get("connected_on") or ""
        import re as _re
        dm = _re.match(r'(\d{1,2})\s+[A-Za-z]{3}\s+(\d{4})', raw_date)
        conn_year = int(dm.group(2)) if dm else None
        if conn_year is None:
            return 0.0  # date unparseable — exclude
        if after_year  is not None and conn_year < int(after_year):
            return 0.0
        if before_year is not None and conn_year >= int(before_year):
            return 0.0

    # Exclusion penalty — covers position, company, location, and name
    for excl in filters.get("exclude_keywords", []):
        excl_l = excl.lower()
        if excl_l in position or excl_l in company or excl_l in location or excl_l in full_name:
            return 0.0

    if weights == 0:
        return 0.0  # No criteria extracted = no match
    return min(score / weights, 1.0)


def filter_connections(connections: list, filters: dict) -> list:
    threshold = filters.get("confidence_threshold", SEARCH_THRESHOLD)
    results = []
    for conn in connections:
        s = score_connection(conn, filters)
        if s >= threshold:
            results.append({**conn, "_score": round(s, 3)})
    results.sort(key=lambda x: x["_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# ICP COMPARISON (enrichment-aware)
# ---------------------------------------------------------------------------

ICP_SYSTEM = """You are a B2B market analyst. The user will describe 2-3 Ideal Customer Profiles (ICPs).
For each ICP, extract filter criteria in the same JSON format as the filter schema.
Return a JSON array, one object per ICP, each with an added "icp_name" string field.

IMPORTANT: Be specific and restrictive with keywords. Each ICP must have meaningful position_keywords and/or position_concepts that clearly differentiate it. Set confidence_threshold to 0.65.
Return ONLY the JSON array."""


def extract_icps(user_message: str) -> list:
    client = get_claude()
    resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=2048,
        system=ICP_SYSTEM,
        messages=[{"role": "user", "content": user_message}]
    )
    result = _parse_json_response(resp.content[0].text)
    return result if isinstance(result, list) else []


def compare_icps(connections: list, icp_descriptions: str, auto_enrich_threshold: int = 50, enrich_fn=None, save_fn=None) -> dict:
    """
    Compare contacts against ICPs.
    - auto_enrich_threshold: if unenriched contacts <= this number, enrich automatically
    - enrich_fn: callable(linkedin_url, company) -> {location, industry, description}
    - save_fn: callable(linkedin_url, industry, description, location) -> None
    Returns dict with keys: results (list of ICP cards), enrichment_performed (bool),
    enrichment_needed (bool), unenriched_count (int), unenriched_urls (list)
    """
    icps = extract_icps(icp_descriptions)

    # Count unenriched contacts
    unenriched = [c for c in connections if not c.get("enriched_at")]
    unenriched_count = len(unenriched)
    enrichment_performed = False

    if unenriched_count > 0 and enrich_fn and save_fn:
        if unenriched_count <= auto_enrich_threshold:
            # Auto-enrich silently
            company_cache = {}
            for c in unenriched:
                url = c.get("linkedin_url", "")
                company = c.get("company", "")
                try:
                    data = enrich_fn(url, company)
                    industry = data.get("industry", "")
                    description = data.get("description", "")
                    location = data.get("location", "")
                    # Use cached company data if individual lookup was weak
                    if (not industry or not description) and company in company_cache:
                        industry = industry or company_cache[company].get("industry", "")
                        description = description or company_cache[company].get("description", "")
                    if company not in company_cache:
                        company_cache[company] = {"industry": industry, "description": description}
                    save_fn(url, industry, description, location)
                    # Update the in-memory contact so scoring uses fresh data
                    c["enriched_industry"] = industry
                    c["enriched_company_desc"] = description
                    c["location"] = location
                    c["enriched_at"] = "just_now"
                except Exception:
                    pass
            enrichment_performed = True
        else:
            # Too many to auto-enrich — ask user
            return {
                "results": [],
                "enrichment_performed": False,
                "enrichment_needed": True,
                "unenriched_count": unenriched_count,
                "unenriched_urls": [c.get("linkedin_url", "") for c in unenriched],
                "icp_descriptions": icp_descriptions,
            }

    results = []
    for icp in icps:
        icp["confidence_threshold"] = max(icp.get("confidence_threshold", ICP_THRESHOLD), ICP_MIN_THRESHOLD)
        matched = filter_connections(connections, icp)
        high_confidence = [m for m in matched if m["_score"] >= ICP_HIGH_THRESHOLD]
        results.append({
            "icp_name": icp.get("icp_name", "ICP"),
            "intent_summary": icp.get("intent_summary", ""),
            "total_matches": len(matched),
            "high_confidence_matches": len(high_confidence),
            "top_companies": _top_values([m["company"] for m in matched[:50]], 5),
            "top_positions": _top_values([m["position"] for m in matched[:50]], 5),
            "sample_contacts": matched[:5],
            "filters_used": icp,
        })
    return {
        "results": results,
        "enrichment_performed": enrichment_performed,
        "enrichment_needed": False,
        "unenriched_count": unenriched_count,
        "unenriched_urls": [],
        "icp_descriptions": icp_descriptions,
    }


def _top_values(values: list, n: int) -> list:
    from collections import Counter
    clean = [v for v in values if v]
    return [item for item, _ in Counter(clean).most_common(n)]


# ---------------------------------------------------------------------------
# RESPONSE SYNTHESIS (for Search mode)
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are a B2B network intelligence assistant.
Given a user's search request and a list of matching LinkedIn connections, write a clear, concise response.

Structure your response as:
1. A brief summary of what you found (1-2 sentences)
2. Key observations about the result set (industries, seniority mix, notable companies)
3. Any caveats or suggestions for refining the search

Be conversational, specific, and helpful. Do not list every contact — the table handles that.
Keep your response under 200 words."""


def synthesise_response(user_message: str, filters: dict, results: list, conversation_history: list) -> str:
    client = get_claude()

    search_mode = filters.get('search_mode', 'new')
    context = f"""User asked: {user_message}

Search intent: {filters.get('intent_summary', '')}
Search mode: {search_mode} (new=fresh list, narrow=filtered subset of prior results, expand=added to prior results)

Results: {len(results)} contacts matched.

Top companies in results: {_top_values([r['company'] for r in results[:100]], 8)}
Top positions in results: {_top_values([r['position'] for r in results[:100]], 8)}
Score distribution: high (>=0.8): {len([r for r in results if r['_score'] >= 0.8])}, medium (0.6-0.8): {len([r for r in results if 0.6 <= r['_score'] < 0.8])}, lower (<0.6): {len([r for r in results if r['_score'] < 0.6])}
"""

    messages = conversation_history[-4:] + [{"role": "user", "content": context}]
    resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=512,
        system=SYNTHESIS_SYSTEM,
        messages=messages
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# ENRICHMENT (online lookup via Tavily)
# ---------------------------------------------------------------------------

def enrich_contact(linkedin_url: str, company_name: str) -> dict:
    """
    Enrich a contact with location, industry, and company description.
    Strategy:
    1. PRIMARY: Fetch the contact's public LinkedIn profile URL via Tavily
    2. FALLBACK: Search for the company + LinkedIn company page
    """
    tavily = get_tavily()
    client = get_claude()
    snippets = ""

    try:
        profile_result = tavily.search(
            query=linkedin_url,
            search_depth="basic",
            max_results=2,
            include_domains=["linkedin.com"]
        )
        snippets = " ".join([r.get("content", "") for r in profile_result.get("results", [])])
    except Exception:
        pass

    if len(snippets.strip()) < 100 and company_name:
        try:
            company_result = tavily.search(
                query=f"{company_name} LinkedIn company page headquarters location industry",
                search_depth="basic",
                max_results=3
            )
            company_snippets = " ".join([r.get("content", "") for r in company_result.get("results", [])])
            snippets = snippets + "\n" + company_snippets
        except Exception:
            pass

    if not snippets.strip():
        return {"industry": "", "description": "", "location": ""}

    try:
        resp = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=256,
            system="""From the search results about a LinkedIn contact and/or their company, extract:
- location: the person's city/region (from their LinkedIn profile if visible), or company HQ as fallback.
  Format rules:
  • US contacts: "City, State, USA" (e.g. "San Francisco, CA, USA" or "Austin, TX, USA")
  • Non-US contacts: "City, Country" (e.g. "London, UK" or "Berlin, Germany" or "Toronto, Canada")
  Use standard 2-letter US state abbreviations. Use common country name (not ISO code).
  Empty string if location cannot be determined.
- industry: the company's industry sector (e.g. "FinTech", "SaaS", "Healthcare", "Consulting"). Empty string if unknown.
- description: one concise sentence describing what the company does. Empty string if unknown.

Return ONLY valid JSON: {"location": "...", "industry": "...", "description": "..."}""",
            messages=[{"role": "user", "content": f"LinkedIn URL: {linkedin_url}\nCompany: {company_name}\n\nSearch results:\n{snippets[:3000]}"}]
        )
        result = _parse_json_response(resp.content[0].text)
        return result if isinstance(result, dict) and "industry" in result else {"industry": "", "description": "", "location": ""}
    except Exception as e:
        return {"industry": "", "description": f"Lookup failed: {str(e)}", "location": ""}


# ---------------------------------------------------------------------------
# DISCOVERY MODE — Sonnet-powered, blends DB search + web research
# ---------------------------------------------------------------------------

DISCOVERY_SYSTEM = """You are a senior B2B business development analyst with deep expertise in markets, job titles, company types, and go-to-market strategy. You have access to the user's LinkedIn network database and can search the web for additional context.

Your role in Discovery mode:
- Act like a thoughtful research partner, not just a search engine
- Proactively surface angles and insights the user hasn't considered
- Ask clarifying questions when they would meaningfully change your analysis
- Use web research to enrich your understanding of industries, job titles, market segments, and geographies
- Blend database findings with market context to give strategic recommendations

You have access to:
- The user's full LinkedIn connections database (provided as context)
- Web search results (provided inline when relevant)
- Conversation history

Response style:
- Conversational but substantive — think out loud with the user
- Use **bold** for key terms and insights
- When you want to propose a web search, format it EXACTLY as: [SEARCH: your search query here]
  Only propose searches that would genuinely add value. Max 2 per response.
- When you want to search the database, format it EXACTLY as: [DB_SEARCH: describe what to find]
  Max 1 per response.
- Ask at most 2 clarifying questions per response, only when the answer would change your analysis significantly
- Keep responses under 400 words unless depth is clearly warranted"""


def discovery_response(
    user_message: str,
    conversation_history: list,
    connections: list,
    db_stats: dict,
    proposed_searches: list = None,  # web search queries approved by user
    proposed_db_search: str = None,  # db search approved by user
) -> dict:
    """
    Discovery mode: Sonnet-powered response that blends DB context, web research, and proactive insights.

    Returns dict with:
    - message: str (the response text, may contain [SEARCH:...] and [DB_SEARCH:...] tags)
    - type: "discovery"
    - proposed_searches: list of search queries extracted from response (need user confirmation)
    - proposed_db_search: str or None (db search query, need user confirmation)
    - web_results: dict of {query: summary} for approved searches already run
    - db_results: list of contacts if db search was approved and run
    - total: int
    """
    client = get_claude()
    tavily = get_tavily()

    # Run any pre-approved web searches
    web_context = ""
    web_results_map = {}
    if proposed_searches:
        for query in proposed_searches:
            try:
                res = tavily.search(query=query, search_depth="basic", max_results=4)
                snippets = "\n".join([
                    f"- {r.get('title', '')}: {r.get('content', '')[:300]}"
                    for r in res.get("results", [])
                ])
                web_results_map[query] = snippets
                web_context += f"\n\n[Web search: '{query}']\n{snippets}"
            except Exception as e:
                web_context += f"\n\n[Web search: '{query}' failed: {str(e)}]"

    # Run pre-approved DB search
    db_results = []
    db_context_str = ""
    if proposed_db_search:
        try:
            filters = extract_filters(proposed_db_search, [])
            db_results = filter_connections(connections, filters)
            top_cos = _top_values([r["company"] for r in db_results[:100]], 8)
            top_pos = _top_values([r["position"] for r in db_results[:100]], 8)
            db_context_str = f"\n\n[DB search: '{proposed_db_search}' → {len(db_results)} contacts found]\nTop companies: {top_cos}\nTop positions: {top_pos}"
        except Exception as e:
            db_context_str = f"\n\n[DB search failed: {str(e)}]"

    # Build system context
    enriched_count = db_stats.get("enriched", 0)
    total_count = db_stats.get("total", 0)
    companies_count = db_stats.get("companies", 0)
    system = DISCOVERY_SYSTEM + f"""

Database context:
- {total_count} total LinkedIn connections
- {companies_count} unique companies
- {enriched_count} contacts enriched with industry/location data ({round(enriched_count / max(total_count, 1) * 100)}% enriched)

Sample of database positions (for context): {_top_values([c.get("position", "") for c in connections[:500] if c.get("position")], 20)}
Sample of database companies: {_top_values([c.get("company", "") for c in connections[:500] if c.get("company")], 20)}
"""

    if web_context:
        system += f"\n\nWeb research results:{web_context}"
    if db_context_str:
        system += f"\n\nDatabase search results:{db_context_str}"

    messages = conversation_history[-12:] + [{"role": "user", "content": user_message}]

    resp = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=1024,
        system=system,
        messages=messages
    )
    message_text = resp.content[0].text.strip()

    # Extract any proposed searches from the response
    new_proposed_searches = re.findall(r'\[SEARCH:\s*([^\]]+)\]', message_text)
    new_proposed_db_search_matches = re.findall(r'\[DB_SEARCH:\s*([^\]]+)\]', message_text)
    new_proposed_db_search = new_proposed_db_search_matches[0] if new_proposed_db_search_matches else None

    # Clean the tags from displayed message
    display_message = re.sub(r'\[SEARCH:\s*[^\]]+\]', '', message_text)
    display_message = re.sub(r'\[DB_SEARCH:\s*[^\]]+\]', '', display_message)
    display_message = display_message.strip()

    return {
        "type": "discovery",
        "message": display_message,
        "proposed_searches": new_proposed_searches,
        "proposed_db_search": new_proposed_db_search,
        "web_results": web_results_map,
        "db_results": db_results[:20],  # preview only
        "total": len(db_results),
        "has_proposals": bool(new_proposed_searches or new_proposed_db_search),
    }


# ---------------------------------------------------------------------------
# GENERAL CHAT (legacy, kept for fallback)
# ---------------------------------------------------------------------------

CHAT_SYSTEM = """You are a B2B network intelligence assistant helping a business development professional
explore and analyse their LinkedIn network.

You help with:
- Searching and filtering contacts by role, company, industry
- Identifying Ideal Customer Profiles (ICPs) from their network
- Understanding what companies or job titles mean
- Suggesting BD strategies based on network composition

Be concise, practical, and focused on business development value."""


def chat_response(user_message: str, conversation_history: list, context: str = "") -> str:
    client = get_claude()
    system = CHAT_SYSTEM
    if context:
        system += f"\n\nCurrent database context: {context}"

    messages = conversation_history[-10:] + [{"role": "user", "content": user_message}]
    resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=512,
        system=system,
        messages=messages
    )
    return resp.content[0].text.strip()
