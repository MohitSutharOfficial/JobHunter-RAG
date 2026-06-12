"""Load job postings from PDF, TXT, Markdown and JSON files into Documents."""

import io
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from job_rag.chunking import chunk_documents
from job_rag.config import Settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json"}


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(
    filename: str,
    data: bytes,
    *,
    title: str | None = None,
    company: str | None = None,
    location: str | None = None,
) -> list[Document]:
    """Parse an uploaded file into one or more job-posting Documents.

    JSON files must contain a list of objects with at least a ``description``
    field, plus optional ``title``, ``company`` and ``location``. Other formats
    are treated as a single posting whose metadata comes from the keyword args.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    if ext == ".json":
        postings = json.loads(data.decode("utf-8"))
        if not isinstance(postings, list):
            raise ValueError("JSON file must contain a list of job postings")
        docs = []
        for i, posting in enumerate(postings):
            description = posting.get("description", "").strip()
            if not description:
                raise ValueError(f"Posting at index {i} has no 'description'")
            docs.append(
                Document(
                    page_content=description,
                    metadata={
                        "title": posting.get("title", "Unknown"),
                        "company": posting.get("company", "Unknown"),
                        "location": posting.get("location", "Unknown"),
                        "source": filename,
                    },
                )
            )
        return docs

    text = _extract_pdf_text(data) if ext == ".pdf" else data.decode("utf-8")
    text = text.strip()
    if not text:
        raise ValueError("File contains no extractable text")

    return [
        Document(
            page_content=text,
            metadata={
                "title": title or Path(filename).stem,
                "company": company or "Unknown",
                "location": location or "Unknown",
                "source": filename,
            },
        )
    ]


def ingest_documents(docs: list[Document], store: VectorStore, settings: Settings) -> int:
    """Chunk documents and add them to the vector store. Returns chunk count."""
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    if chunks:
        store.add_documents(chunks)
    return len(chunks)
