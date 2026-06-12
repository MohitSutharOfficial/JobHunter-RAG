from langchain_core.documents import Document

from job_rag.retrieval import answer_question, rank_jobs


def _seed(store):
    store.add_documents(
        [
            Document(
                page_content="Senior Python engineer with FastAPI and ML experience.",
                metadata={"title": "Python Dev", "company": "Acme", "location": "Remote"},
            ),
            Document(
                page_content="React and TypeScript frontend developer.",
                metadata={"title": "Frontend Dev", "company": "PixelSoft", "location": "NYC"},
            ),
        ]
    )


def test_rank_jobs_returns_deduplicated_sorted_matches(store):
    _seed(store)
    matches = rank_jobs("Python machine learning resume", store, k=2)
    assert len(matches) == 2
    assert matches[0].score <= matches[1].score
    titles = {m.title for m in matches}
    assert titles == {"Python Dev", "Frontend Dev"}


def test_answer_question_without_llm_returns_sources(settings, store):
    _seed(store)
    result = answer_question("Which jobs need Python?", store, settings)
    assert result.sources
    assert "LLM is not configured" in result.answer


def test_answer_question_empty_store(settings, store):
    result = answer_question("Anything?", store, settings)
    assert result.sources == []
    assert "No job postings" in result.answer
