import json

import pytest

from job_rag.ingestion import ingest_documents, load_documents


def test_load_json_postings():
    payload = json.dumps(
        [
            {"title": "Python Dev", "company": "Acme", "location": "Remote", "description": "Build APIs."},
            {"title": "ML Engineer", "company": "DataWorks", "description": "Train models."},
        ]
    ).encode()
    docs = load_documents("jobs.json", payload)
    assert len(docs) == 2
    assert docs[0].metadata["title"] == "Python Dev"
    assert docs[1].metadata["location"] == "Unknown"


def test_load_txt_posting_with_metadata():
    docs = load_documents("posting.txt", b"Great Python job.", title="Py Dev", company="Acme")
    assert len(docs) == 1
    assert docs[0].metadata["title"] == "Py Dev"
    assert docs[0].page_content == "Great Python job."


def test_load_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_documents("jobs.docx", b"data")


def test_load_json_missing_description_raises():
    with pytest.raises(ValueError, match="no 'description'"):
        load_documents("jobs.json", json.dumps([{"title": "X"}]).encode())


def test_ingest_documents_indexes_chunks(settings, store):
    docs = load_documents("posting.txt", ("python " * 100).encode(), title="Py Dev")
    count = ingest_documents(docs, store, settings)
    assert count >= 1
    results = store.similarity_search("python", k=1)
    assert results
    assert results[0].metadata["title"] == "Py Dev"
