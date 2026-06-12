"""Chroma vector store factory."""

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from job_rag.config import Settings


def get_vectorstore(settings: Settings, embeddings: Embeddings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )


def get_in_memory_vectorstore(embeddings: Embeddings, collection_name: str = "test") -> Chroma:
    """Ephemeral store, used in tests."""
    return Chroma(collection_name=collection_name, embedding_function=embeddings)
