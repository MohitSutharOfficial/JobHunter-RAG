"""Embeddings factory: Gemini embeddings or deterministic fakes for dev/tests."""

from langchain_core.embeddings import Embeddings

from job_rag.config import Settings


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.use_fake_embeddings or not settings.google_api_key:
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=384)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.google_api_key
    )
