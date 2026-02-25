import os
import json
import anthropic
from tavily import TavilyClient

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TAVILY_KEY = os.environ.get("TAVILY_API_KEY", "")

def get_claude():
    return anthropic.Anthropic(api_key=ANTHROPIC_KEY)

def get_tavily():
    return TavilyClient(api_key=TAVILY_KEY)


# ---------------------------------------------------------------------------
# SEMANTIC FILTER
# ---------------------------------------------------------------------------

FILTER_SYSTEM = """You are a B2B network intelligence assistant helping analyse a LinkedIn connections database.

Your job: given a natural language search request, extract structured filter criteria and return ONLY valid JSON.

The database has these fields:
- first_name, last_name
- company (column E in LinkedIn export)
- position (column F in LinkedIn export)
- connected_on
- enriched_industry (may be empty)
- enriched_company_desc (may be empty)

Return JSON with this exact schema:
{
  "intent_summary": "1-2 sentence plain English summary of what the user is looking for",
  "company_keywords": ["list", "of", "company", "name", "fragments"],
  "position_keywords": ["list", "of", "job", "title", "keywords"],
  "position_concepts": ["broader role concepts to match against, e.g. 'revenue leader', 'technical founder'"],
  "seniority_levels": ["C-suite", "VP", "Director", "Manager", "Individual Contributor", "Founder"],
  "industries": ["list of industries if mentioned"],
  "exclude_keywords": ["words that should NOT appear in company or position"],
  "confidence_threshold": 0.6
}

For position_concepts, think broadly. E.g. "head of sales" should match: VP Sales, Chief Revenue Officer, Director of Sales, Head of Revenue, Sales Director, Commercial Director etc.
For seniority_levels, only include if the user specifies seniority.
All arrays can be empty [].
Return ONLY the JSON object, no markdown, no explanation."""


def extract_filters(user_message: str, conversation_history: list) -> dict:
    client = get_claude()
    messages = conversation_history[-6:] + [{"role": "user", "content": user_message}]

    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=FILTER_SYSTEM,
        messages=messages
    )
    raw = resp.content[0].text.strip()
    # Strip markdown code blocks if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------

def score_connection(conn: dict, filters: dict) -> float:
    """
    Score a connection 0.0-1.0 against extracted filters.
    Returns score and populates no external calls.
    """
    score = 0.0
    weights = 0.0
    position = (conn.get("position") or "").lower()
    company = (conn.get("company") or "").lower()
    industry = (conn.get("enriched_industry") or "").lower()
    desc = (conn.get("enriched_company_desc") or "").lower()

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
                score += 1.0
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

    # Exclusion penalty
    for excl in filters.get("exclude_keywords", []):
        if excl.lower() in position or excl.lower() in company:
            return 0.0

    if weights == 0:
        return 0.5  # No criteria = everything passes
    return min(score / weights, 1.0)


def filter_connections(connections: list[dict], filters: dict) -> list[dict]:
    threshold = filters.get("confidence_threshold", 0.5)
    results = []
    for conn in connections:
        s = score_connection(conn, filters)
        if s >= threshold:
            results.append({**conn, "_score": round(s, 3)})
    results.sort(key=lambda x: x["_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# ICP COMPARISON (quick, local only)
# ---------------------------------------------------------------------------

ICP_SYSTEM = """You are a B2B market analyst. The user will describe 2-3 Ideal Customer Profiles (ICPs).
For each ICP, extract filter criteria in the same JSON format as the filter schema.
Return a JSON array, one object per ICP, each with an added "icp_name" string field.
Return ONLY the JSON array."""

def extract_icps(user_message: str) -> list[dict]:
    client = get_claude()
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2048,
        system=ICP_SYSTEM,
        messages=[{"role": "user", "content": user_message}]
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)

def compare_icps(connections: list[dict], icp_descriptions: str) -> list[dict]:
    icps = extract_icps(icp_descriptions)
    results = []
    for icp in icps:
        matched = filter_connections(connections, icp)
        high_confidence = [m for m in matched if m["_score"] >= 0.7]
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
    return results

def _top_values(values: list, n: int) -> list:
    from collections import Counter
    clean = [v for v in values if v]
    return [item for item, _ in Counter(clean).most_common(n)]


# ---------------------------------------------------------------------------
# RESPONSE SYNTHESIS
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM = """You are a B2B network intelligence assistant.
Given a user's search request and a list of matching LinkedIn connections, write a clear, concise response.

Structure your response as:
1. A brief summary of what you found (1-2 sentences)
2. Key observations about the result set (industries, seniority mix, notable companies)
3. Any caveats or suggestions for refining the search

Be conversational, specific, and helpful. Do not list every contact — the table handles that.
Keep your response under 200 words."""

def synthesise_response(
    user_message: str,
    filters: dict,
    results: list[dict],
    conversation_history: list
) -> str:
    client = get_claude()

    context = f"""User asked: {user_message}

Search intent: {filters.get('intent_summary', '')}

Results: {len(results)} contacts matched.

Top companies in results: {_top_values([r['company'] for r in results[:100]], 8)}
Top positions in results: {_top_values([r['position'] for r in results[:100]], 8)}
Score distribution: high (>=0.8): {len([r for r in results if r['_score'] >= 0.8])}, medium (0.6-0.8): {len([r for r in results if 0.6 <= r['_score'] < 0.8])}, lower (<0.6): {len([r for r in results if r['_score'] < 0.6])}
"""

    messages = conversation_history[-4:] + [{"role": "user", "content": context}]
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=SYNTHESIS_SYSTEM,
        messages=messages
    )
    return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# ENRICHMENT (online lookup via Tavily)
# ---------------------------------------------------------------------------

def enrich_company(company_name: str) -> dict:
    """Look up a company online and return industry, description, and HQ location."""
    tavily = get_tavily()
    try:
        result = tavily.search(
            query=f"{company_name} company industry headquarters location what do they do",
            search_depth="basic",
            max_results=3
        )
        snippets = " ".join([r.get("content", "") for r in result.get("results", [])])

        client = get_claude()
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=256,
            system="""Extract industry, a 1-sentence company description, and headquarters city/location from the text.
Return JSON: {"industry": "...", "description": "...", "location": "City, Country"}.
For location use the format 'City, Country' or just 'City' if country is obvious. Use empty string if unknown.
Return ONLY JSON.""",
            messages=[{"role": "user", "content": f"Company: {company_name}\n\nSearch results:\n{snippets[:2000]}"}]
        )
        raw = resp.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception as e:
        return {"industry": "", "description": f"Lookup failed: {str(e)}", "location": ""}


# ---------------------------------------------------------------------------
# GENERAL CHAT (for follow-up questions, clarifications)
# ---------------------------------------------------------------------------

CHAT_SYSTEM = """You are a B2B network intelligence assistant helping a business development professional
explore and analyse their LinkedIn network.

You help with:
- Searching and filtering contacts by role, company, industry
- Identifying Ideal Customer Profiles (ICPs) from their network
- Understanding what companies or job titles mean
- Suggesting BD strategies based on network composition

Be concise, practical, and focused on business development value.
When the user describes a search, acknowledge it and confirm what you understood before searching.
When asked about ICPs, offer a quick estimate first, then ask if they want a deep search."""

def chat_response(user_message: str, conversation_history: list, context: str = "") -> str:
    client = get_claude()
    system = CHAT_SYSTEM
    if context:
        system += f"\n\nCurrent database context: {context}"

    messages = conversation_history[-10:] + [{"role": "user", "content": user_message}]
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=system,
        messages=messages
    )
    return resp.content[0].text.strip()
