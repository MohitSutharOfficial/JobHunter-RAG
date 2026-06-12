---
title: Job Hunting RAG
emoji: "\U0001F3AF"
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Job Hunting RAG

End-to-end Retrieval-Augmented Generation (RAG) service for job hunting.
Ingest job postings, upload your resume, then ask questions and get ranked job
matches with explanations.

**Stack:** Python 3.11, FastAPI, LangChain, Chroma (vector store), Groq (LLM, free tier), Gemini (embeddings + fallback LLM, free tier).

## Features

- **Ingestion** of job postings from PDF, TXT, Markdown and JSON files
- **Resume upload** for personalized matching
- **Question answering** over ingested jobs (RAG)
- **Resume-to-job matching** with ranked results and optional LLM explanations
- Runs fully offline in tests via deterministic fake embeddings

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY and GOOGLE_API_KEY (both free)
uvicorn job_rag.api:app --reload --app-dir src
```

API docs: http://localhost:8000/docs

## Quickstart (Docker)

```bash
docker build -t job-rag .
docker run -p 8000:8000 --env-file .env -v $(pwd)/data:/app/data job-rag
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | _(empty)_ | Groq API key (free at https://console.groq.com), preferred LLM |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq chat model |
| `GOOGLE_API_KEY` | _(empty)_ | Google AI Studio key (free at https://aistudio.google.com/apikey), used for embeddings and as fallback LLM |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini chat model (fallback LLM) |
| `EMBEDDING_MODEL` | `models/text-embedding-004` | Gemini embedding model |
| `CHROMA_DIR` | `./data/chroma` | Chroma persistence directory |
| `COLLECTION_NAME` | `job_postings` | Chroma collection name |
| `CHUNK_SIZE` | `800` | Chunk size (characters) |
| `CHUNK_OVERLAP` | `120` | Chunk overlap (characters) |
| `TOP_K` | `5` | Default number of retrieved chunks |
| `USE_FAKE_EMBEDDINGS` | `false` | Use deterministic fake embeddings (dev/tests) |

## API examples

Ingest job postings (JSON file with a list of postings):

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@examples/jobs.json"
```

Ingest a single posting from PDF/TXT/MD:

```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@posting.pdf" \
  -F "title=Senior ML Engineer" -F "company=Acme" -F "location=Remote"
```

Upload your resume:

```bash
curl -X POST http://localhost:8000/resume -F "file=@resume.pdf"
```

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which jobs best match my Python and ML experience?"}'
```

Get ranked matches for your resume:

```bash
curl -X POST http://localhost:8000/match \
  -H 'Content-Type: application/json' \
  -d '{"top_k": 5, "explain": true}'
```

## JSON posting format

```json
[
  {
    "title": "Senior Python Engineer",
    "company": "Acme Corp",
    "location": "Remote",
    "description": "We are looking for a Python engineer with ML experience..."
  }
]
```

## Web UI

A professional single-page web app is served at the root URL (`/`):

- Drag-and-drop ingestion of job postings
- Resume upload (PDF/TXT)
- Grounded Q&A with cited sources
- Ranked job matches with fit scores and optional AI explanations

Start the server and open http://localhost:8000

## Deployment

On every push to `main`, CI builds the Docker image and pushes it to the
**GitLab Container Registry**:

```
registry.gitlab.com/my-projects6853305/job-hunting-rag:latest
```

Run it anywhere Docker is available (VPS, Cloud Run, Fly.io, Render, etc.):

```bash
docker login registry.gitlab.com
docker run -d -p 8000:8000 \
  -e GROQ_API_KEY=gsk_... \
  -e GOOGLE_API_KEY=AIza... \
  -v $(pwd)/data:/app/data \
  registry.gitlab.com/my-projects6853305/job-hunting-rag:latest
```

The `data/` volume persists the Chroma index and uploaded resume across restarts.

### Free deployment: Hugging Face Spaces (recommended)

The app deploys for **free** to a Docker-based [Hugging Face Space](https://huggingface.co/spaces).
The YAML front matter at the top of this README configures the Space
(`sdk: docker`, `app_port: 8000`).

**One-time setup:**

1. Create a free account at https://huggingface.co and create a new **Space**
   (type: **Docker**, blank template), e.g. `your-username/job-hunting-rag`.
2. Create a **write** access token at https://huggingface.co/settings/tokens.
3. In GitLab, go to **Settings > CI/CD > Variables** and add:
   - `HF_TOKEN` (masked) - the write token
   - `HF_USERNAME` - your Hugging Face username
   - `HF_SPACE` (optional) - the Space name, defaults to `job-hunting-rag`
4. In the Space settings on Hugging Face, add **secrets** named
   `GROQ_API_KEY` (from https://console.groq.com) and `GOOGLE_API_KEY`
   (from https://aistudio.google.com/apikey). Both have free tiers.

Every merge to `main` then triggers the `deploy-huggingface` CI job, which
pushes the code to the Space. It builds and goes live at:

```
https://huggingface.co/spaces/<HF_USERNAME>/job-hunting-rag
```

> **Note:** free Spaces have ephemeral disk and sleep after inactivity, so the
> Chroma index resets when the Space restarts. Re-ingest postings after a
> restart, or use a hosted vector DB for persistence.

### Deploy to Cloudflare (Containers + Workers, paid plan)

The app deploys to **Cloudflare Containers**: a lightweight Worker
(`worker/index.js`) routes all traffic to the FastAPI container defined by the
`Dockerfile`, configured in `wrangler.jsonc`.

**One-time setup:**

1. You need a Cloudflare account with the **Workers Paid** plan (required for Containers).
2. Create an API token at Cloudflare dashboard with the *Edit Cloudflare Workers* template.
3. In GitLab, go to **Settings > CI/CD > Variables** and add (masked):
   - `CLOUDFLARE_API_TOKEN`
   - `CLOUDFLARE_ACCOUNT_ID`
4. Set the API keys as Worker secrets (from your machine):
   ```bash
   npm install
   npx wrangler secret put GROQ_API_KEY
   npx wrangler secret put GOOGLE_API_KEY
   ```

After that, every merge to `main` triggers the `deploy-cloudflare` CI job,
which builds the container and deploys it. Your app will be live at:

```
https://job-hunting-rag.<your-subdomain>.workers.dev
```

You can also deploy manually from your machine:

```bash
npx wrangler login
npx wrangler deploy
```

> **Note:** Cloudflare container instances sleep after 15 minutes of
> inactivity and their disk is ephemeral, so the Chroma index resets when the
> instance recycles. For persistent production data, re-ingest on startup or
> point the app at a hosted vector DB.

## Beta launch checklist (clone & deploy from GitHub)

The repo is fully portable. To launch the beta from GitHub:

1. **Clone and push to GitHub:**
   ```bash
   git clone https://gitlab.com/my-projects6853305/job-hunting-rag.git
   cd job-hunting-rag
   git remote add github https://github.com/<you>/job-hunting-rag.git
   git push github main
   ```
2. **Configure GitHub Actions** (Settings > Secrets and variables > Actions):
   - Secret `HF_TOKEN` - Hugging Face write token (free)
   - Variable `HF_USERNAME` - your Hugging Face username
3. **Create the free Hugging Face Space** (type: Docker, name `job-hunting-rag`)
   and add Space secrets `GROQ_API_KEY` and `GOOGLE_API_KEY` (both free tiers).
4. Push to `main` on GitHub - the `CI & Deploy` workflow lints, tests and
   deploys automatically. Your beta is live at
   `https://huggingface.co/spaces/<HF_USERNAME>/job-hunting-rag`.

**Or self-host with one command** on any machine with Docker:

```bash
cp .env.example .env   # add free GROQ_API_KEY + GOOGLE_API_KEY
docker compose up -d   # app on http://localhost:8000, data persisted in ./data
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

Tests use deterministic fake embeddings and an in-memory Chroma collection, so
no API key or network access is required.
