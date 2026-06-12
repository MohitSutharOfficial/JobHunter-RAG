from langchain_core.documents import Document

from job_rag.chunking import chunk_documents


def test_chunk_documents_splits_long_text():
    doc = Document(page_content="word " * 500, metadata={"title": "Job A"})
    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_chunk_documents_preserves_metadata():
    doc = Document(page_content="short text", metadata={"title": "Job A", "company": "Acme"})
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].metadata["title"] == "Job A"
    assert chunks[0].metadata["company"] == "Acme"
