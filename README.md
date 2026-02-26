# Network Scanner — B2B Network Intelligence

A private B2B network intelligence tool that lets you upload your LinkedIn connections export, search and filter contacts using natural language (AI-powered), enrich contacts with company/industry/location data via Tavily, and run ICP (Ideal Customer Profile) matching.

## Modes

| Mode | Description |
|------|-------------|
| **Search** | Natural language search across your network (Claude Haiku) |
| **ICP** | Define 2-3 Ideal Customer Profiles, score your network against them, auto-enrich as needed |
| **Discovery** | Deep research mode — blends DB search, live web research (Tavily), and strategic insights (Claude Sonnet) |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/carlhinrichsen/network-scanner.git
cd network-scanner
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API key (claude-haiku + claude-sonnet) |
| `TAVILY_API_KEY` | ✅ | Tavily search API key (enrichment + Discovery mode) |
| `GOOGLE_CLIENT_ID` | ✅ | Google OAuth 2.0 client ID |
| `GOOGLE_CLIENT_SECRET` | ✅ | Google OAuth 2.0 client secret |
| `APP_BASE_URL` | ✅ | Your deployed app URL (e.g. `https://your-app.onrender.com`) |
| `ADMIN_EMAIL` | ✅ | Your Google email — gets admin access automatically |
| `APPROVED_DOMAINS` | optional | Comma-separated domains auto-approved on login (e.g. `yourcompany.com`) |
| `DB_PATH` | optional | SQLite DB path (default: `data/connections.db`) |

### 3. Google OAuth setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials
2. Create an OAuth 2.0 Client ID (Web application)
3. Add authorized redirect URI: `{APP_BASE_URL}/api/auth/google/callback`
4. Copy Client ID and Secret to your `.env`

### 4. Run locally

```bash
uvicorn main:app --reload --port 8000
```

App will be available at `http://localhost:8000`.

---

## Deployment (Render)

The repo includes `render.yaml` for one-click Render deployment.

1. Push to GitHub
2. Connect repo in Render dashboard
3. Add a **Persistent Disk** (mount at `/data`, 1 GB)
4. Set all env vars in Render dashboard (they are NOT in `render.yaml` for security)
5. Set `DB_PATH=/data/connections.db` in Render env vars

### Rotating API keys

If an API key is compromised:
1. Revoke the old key in the respective dashboard (Anthropic / Tavily / Google Cloud)
2. Generate a new key
3. Update the env var in Render → service will redeploy automatically
4. The `.env.example` file contains **placeholder values only** — never commit real keys

---

## API endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/auth/google/login` | — | Start Google OAuth flow |
| GET | `/api/auth/google/callback` | — | OAuth callback |
| POST | `/api/auth/logout` | session | Logout |
| GET | `/api/auth/me` | — | Check session status |
| GET | `/api/healthz` | — | Health check + contact count |
| GET | `/api/stats` | session | DB statistics |
| POST | `/api/upload` | session | Upload LinkedIn CSV (persists to DB) |
| POST | `/api/guest/upload` | — | Upload LinkedIn CSV (in-memory only) |
| POST | `/api/chat` | session/guest | Search / ICP / Discovery |
| POST | `/api/enrich` | session | Enrich contacts via Tavily |
| POST | `/api/export` | — | Export results as CSV |
| GET | `/api/admin/users` | admin | List all users |
| GET | `/api/admin/pending` | admin | List pending approvals |
| POST | `/api/admin/approve/{id}` | admin | Approve a user |
| POST | `/api/admin/revoke/{id}` | admin | Revoke a user |

---

## Data flow

1. User logs in via Google OAuth
2. Upload LinkedIn connections CSV (Settings → Export Data → Connections)
3. Run natural language searches in Search mode
4. Run ICP matching in ICP mode (optionally enrich first)
5. Use Discovery mode for deeper research with live web context
6. Export filtered results as CSV

---

## Architecture

```
main.py          FastAPI app, routes, OAuth, session management
auth.py          User/session CRUD, auto-approval logic
database.py      Connections CRUD, SQLite
ai_engine.py     Claude + Tavily integrations, scoring, all AI functions
csv_parser.py    LinkedIn CSV parsing
static/
  index.html     Single-file frontend (HTML + CSS + JS)
  logo.png       App logo
  favicon.png    Browser favicon
```
