"""Retrieval, question answering and resume-to-job matching."""

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from job_rag.config import Settings

ANSWER_PROMPT = """You are a helpful job-hunting assistant. Use ONLY the job postings
in the context below to answer the question. If a resume is provided, tailor the
answer to the candidate's experience. Cite job titles and companies explicitly.
If the context does not contain the answer, say so.

Format your response with clear structure:
- Use **bold** for key job titles and company names
- Use bullet points for listing multiple items
- Be specific about skill matches when a resume is provided

# Job postings context
{context}

# Candidate resume (may be empty)
{resume}

# Question
{question}
"""

EXPLAIN_PROMPT = """You are a career coach. Given the candidate resume and a job posting,
provide a concise analysis in 2-3 sentences. Be specific about:
1. Which candidate skills directly match the job requirements
2. What skills or experience the candidate is missing
3. An overall verdict (strong match / moderate match / weak match)

Matched skills: {matched_skills}
Missing skills: {missing_skills}

# Resume
{resume}

# Job posting: {title} at {company} ({location})
{description}
"""


@dataclass
class JobMatch:
    title: str
    company: str
    location: str
    score: float
    snippet: str
    explanation: str | None = None
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    match_percent: int = 0


@dataclass
class Answer:
    answer: str
    sources: list[dict] = field(default_factory=list)


def retrieve(query: str, store: VectorStore, k: int) -> list[tuple[Document, float]]:
    """Return (document, distance) pairs; lower distance means more similar."""
    return store.similarity_search_with_score(query, k=k)


def rank_jobs(
    resume_text: str, store: VectorStore, k: int,
    resume_skills: list[str] | None = None,
    settings: Settings | None = None,
) -> list[JobMatch]:
    """Rank job postings against a resume using vector similarity + skill overlap.

    Deduplicates chunks per (title, company), keeping each job's best score.
    If resume_skills are provided, computes skill overlap per job.
    """
    from job_rag.skills import (
        compute_skill_overlap,
        extract_skills_heuristic,
        extract_skills_with_llm,
    )

    results = retrieve(resume_text, store, k=max(k * 3, k))
    best: dict[tuple[str, str], JobMatch] = {}
    job_texts: dict[tuple[str, str], str] = {}

    for doc, distance in results:
        meta = doc.metadata
        key = (meta.get("title", "Unknown"), meta.get("company", "Unknown"))
        match = JobMatch(
            title=key[0],
            company=key[1],
            location=meta.get("location", "Unknown"),
            score=float(distance),
            snippet=doc.page_content[:500],
        )
        if key not in best or match.score < best[key].score:
            best[key] = match
            job_texts[key] = doc.page_content

    ranked = sorted(best.values(), key=lambda m: m.score)[:k]

    # Skill overlap analysis
    if resume_skills:
        use_llm = settings and has_llm(settings)
        llm = _get_llm(settings) if use_llm else None

        for m in ranked:
            key = (m.title, m.company)
            job_text = job_texts.get(key, m.snippet)
            if llm:
                job_skills = extract_skills_with_llm(job_text, llm)
            else:
                job_skills = extract_skills_heuristic(job_text)

            overlap = compute_skill_overlap(resume_skills, job_skills)
            m.matched_skills = overlap["matched"]
            m.missing_skills = overlap["missing"]
            m.extra_skills = overlap["extra"]
            m.match_percent = overlap["match_percent"]

    return ranked


def has_llm(settings: Settings) -> bool:
    return bool(settings.groq_api_key or settings.google_api_key)


def _get_llm(settings: Settings):
    if settings.groq_api_key:
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0)

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model, google_api_key=settings.google_api_key, temperature=0
    )


def _format_context(results: list[tuple[Document, float]]) -> str:
    parts = []
    for doc, _ in results:
        meta = doc.metadata
        parts.append(
            f"- {meta.get('title', 'Unknown')} at {meta.get('company', 'Unknown')} "
            f"({meta.get('location', 'Unknown')}):\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def answer_question(
    question: str, store: VectorStore, settings: Settings,
    resume_text: str = "", k: int | None = None,
) -> Answer:
    """RAG answer: retrieve relevant postings and generate a grounded answer."""
    results = retrieve(question, store, k or settings.top_k)
    sources = [
        {
            "title": doc.metadata.get("title", "Unknown"),
            "company": doc.metadata.get("company", "Unknown"),
            "location": doc.metadata.get("location", "Unknown"),
            "score": float(score),
        }
        for doc, score in results
    ]
    if not results:
        return Answer(answer="No job postings have been ingested yet.", sources=[])

    if not has_llm(settings):
        return Answer(
            answer=(
                "LLM is not configured (set GROQ_API_KEY or GOOGLE_API_KEY). "
                "Returning retrieved matches only."
            ),
            sources=sources,
        )

    llm = _get_llm(settings)
    prompt = ANSWER_PROMPT.format(
        context=_format_context(results),
        resume=resume_text or "(none provided)",
        question=question,
    )
    response = llm.invoke(prompt)
    return Answer(answer=response.content, sources=sources)


def explain_match(
    match: JobMatch, resume_text: str, settings: Settings,
) -> str:
    llm = _get_llm(settings)
    prompt = EXPLAIN_PROMPT.format(
        resume=resume_text,
        title=match.title,
        company=match.company,
        location=match.location,
        description=match.snippet,
        matched_skills=", ".join(match.matched_skills) or "none identified",
        missing_skills=", ".join(match.missing_skills) or "none identified",
    )
    return llm.invoke(prompt).content
