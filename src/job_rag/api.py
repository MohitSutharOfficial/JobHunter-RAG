"""FastAPI application exposing the job-hunting RAG service and web UI."""

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from job_rag.config import Settings, get_settings
from job_rag.embeddings import get_embeddings
from job_rag.ingestion import load_documents, ingest_documents
from job_rag.retrieval import answer_question, explain_match, has_llm, rank_jobs
from job_rag.vectorstore import get_vectorstore

app = FastAPI(title="Job Hunting RAG", version="0.1.0")


@lru_cache
def _store():
    settings = get_settings()
    return get_vectorstore(settings, get_embeddings(settings))


def _read_resume(settings: Settings) -> str:
    path = Path(settings.resume_path)
    return path.read_text(encoding="utf-8") if path.exists() else ""


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    use_resume: bool = True


class MatchRequest(BaseModel):
    top_k: int = 5
    explain: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    company: str | None = Form(None),
    location: str | None = Form(None),
) -> dict:
    settings = get_settings()
    data = await file.read()
    try:
        docs = load_documents(
            file.filename or "upload.txt", data, title=title, company=company, location=location
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    chunks = ingest_documents(docs, _store(), settings)
    return {"postings": len(docs), "chunks_indexed": chunks}


@app.post("/resume")
async def upload_resume(file: UploadFile = File(...)) -> dict:
    settings = get_settings()
    data = await file.read()
    filename = file.filename or "resume.txt"
    if filename.lower().endswith(".pdf"):
        from job_rag.ingestion import _extract_pdf_text

        text = _extract_pdf_text(data)
    else:
        text = data.decode("utf-8")
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Resume contains no extractable text")
    path = Path(settings.resume_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"characters": len(text)}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    settings = get_settings()
    resume_text = _read_resume(settings) if request.use_resume else ""
    result = answer_question(
        request.question, _store(), settings, resume_text=resume_text, k=request.top_k
    )
    return {"answer": result.answer, "sources": result.sources}


@app.post("/match")
def match(request: MatchRequest) -> dict:
    settings = get_settings()
    resume_text = _read_resume(settings)
    if not resume_text:
        raise HTTPException(status_code=400, detail="Upload a resume first via POST /resume")
    matches = rank_jobs(resume_text, _store(), request.top_k)
    if request.explain and has_llm(settings):
        for m in matches:
            m.explanation = explain_match(m, resume_text, settings)
    return {"matches": [asdict(m) for m in matches]}


# Serve the web UI (must be mounted last so API routes take precedence)
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")
