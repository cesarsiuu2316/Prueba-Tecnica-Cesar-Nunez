"""Ingestion for the MineCatalog corpus: load files and clean their text.

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
from pypdf import PdfReader

load_dotenv()
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
SUPPORTED = {".txt", ".md", ".json", ".pdf"}


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
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)       # collapse spaces/tabs within a line
    text = re.sub(r" *\n", "\n", text)         # drop trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse blank-line runs
    return text.strip()


def main() -> None:
    """Load and clean the corpus, then print a short summary."""
    documents = clean_documents(load_documents())
    print(f"Loaded {len(documents)} documents from {DOCS_DIR}/")
    for doc in documents:
        print(f"  {doc.metadata['source']}: {len(doc.page_content)} chars")


if __name__ == "__main__":
    main()
