"""Ingestion for corpus: load files and clean their text.

Reads every supported file in a directory (.txt, .md, .json, .pdf) into
langchain_core Documents and normalizes the text. Chunking and embedding are
handled in later steps.
"""

import os
import re
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
SUPPORTED = {".txt", ".md", ".json", ".pdf"}
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def _read_file(path: Path) -> str:
    """Extract raw text from one file (PDFs via pypdf, everything else as text)."""
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8")

    pages = []
    for page in PdfReader(str(path)).pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def load_documents(docs_dir: Path | str = DOCS_DIR) -> list[Document]:
    """Load every supported file in a directory into langchain Documents."""
    docs_dir = Path(docs_dir)

    documents = []
    for path in sorted(docs_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        text = _read_file(path)
        documents.append(Document(page_content=text, metadata={"source": path.name}))
    return documents


def clean_documents(documents: list[Document]) -> list[Document]:
    """Normalize each document's text in place (NFKC + tidy whitespace)."""
    for doc in documents:
        doc.page_content = _clean_text(doc.page_content)
    return documents


def _clean_text(text: str) -> str:
    """Fix unicode artifacts (e.g. the 'ﬁ' ligature) and collapse stray whitespace."""
    text = unicodedata.normalize("NFKC", text) # 
    text = text.replace("\r\n", "\n").replace("\r", "\n") # unify newlines
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces/tabs within a line
    text = re.sub(r" *\n", "\n", text)         # drop trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse blank-line runs
    return text.strip()


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping, size-bounded chunks for embedding.

    A recursive splitter breaks on natural boundaries first (paragraphs, then
    lines, then sentences), keeping each chunk coherent. Existing metadata
    (e.g. ``source``) is carried over to every chunk automatically.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def main() -> None:
    """Load, clean, and chunk the corpus, then print a short summary."""
    documents = clean_documents(load_documents())
    chunks = chunk_documents(documents)

    counts = {}

    for chunk in chunks:
        source = chunk.metadata["source"]
        counts[source] = counts.get(source, 0) + 1

    print("First 5 chunks:")
    for i, chunk in enumerate(chunks[9:14]):
        print(f"  {i+1}. {chunk.metadata['source']}: {len(chunk.page_content)} chars")

    print(f"Loaded {len(documents)} documents, produced {len(chunks)} chunks:")
    for source in sorted(counts):
        print(f"  {source}: {counts[source]} chunks")


if __name__ == "__main__":
    main()
