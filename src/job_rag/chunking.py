"""Document chunking utilities."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_splitter(
    chunk_size: int = 800, chunk_overlap: int = 120,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(
    docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Document]:
    """Split documents into overlapping chunks, preserving metadata."""
    splitter = make_splitter(chunk_size, chunk_overlap)
    return splitter.split_documents(docs)
