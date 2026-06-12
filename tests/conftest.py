import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from job_rag.config import Settings
from job_rag.vectorstore import get_in_memory_vectorstore


@pytest.fixture
def settings() -> Settings:
    return Settings(
        use_fake_embeddings=True,
        groq_api_key="",
        google_api_key="",
        chunk_size=200,
        chunk_overlap=40,
    )


@pytest.fixture
def store(request):
    embeddings = DeterministicFakeEmbedding(size=384)
    return get_in_memory_vectorstore(embeddings, collection_name=f"test_{request.node.name}")
