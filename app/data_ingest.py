"""Ingestion for corpus: load files, clean their text, and chunk by file type.

Reads every supported file in a directory (.txt, .md, .json, .pdf) into
langchain_core Documents, normalizes the text, and splits it into chunks using a
strategy that fits each format instead of flattening everything into plain text:

- ``.md``   -> split on the ``#`` title (each error stays whole) and lift the
              title, error code, category and keywords into metadata tags, then
              cap each section by size.
- ``.json`` -> one Document per record, with the record's key fields lifted into
              metadata tags (id, categoría, palabras clave, ...).
- ``.txt`` / ``.pdf`` -> a recursive character splitter on natural boundaries.

Embedding and storage are handled in later steps.
"""

import json
import os
import re
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

load_dotenv()
DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
SUPPORTED = {".txt", ".md", ".json", ".pdf"}
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_file(path: Path) -> str:
    """Extract raw text from one file (PDFs via pypdf, everything else as text)."""
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8")

    pages = []
    for page in PdfReader(str(path)).pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def load_documents(docs_dir: Path | str = DOCS_DIR) -> list[Document]:
    """Load every supported file in a directory into langchain Documents.

    Most files become a single raw-text Document tagged with its ``format`` so
    the right chunker can be chosen later. JSON is the exception: it is parsed
    here into one Document per record, because each record is already a natural,
    self-contained chunk with useful fields to keep as metadata.
    """
    docs_dir = Path(docs_dir)

    documents = []
    for path in sorted(docs_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED:
            continue

        if suffix == ".json":
            documents.extend(_load_json_records(path))
        else:
            text = _read_file(path)
            metadata = {"source": path.name, "format": suffix.lstrip(".")}
            documents.append(Document(page_content=text, metadata=metadata))
    return documents


def _load_json_records(path: Path) -> list[Document]:
    """Load a JSON corpus file as one Document per record.

    The sample JSON is an object with a ``contenido`` list of error records, each
    a self-contained unit (id, título, causas, solución, palabras clave). Those
    records are already natural chunks, so we render each one into readable text
    and lift its key fields into metadata tags instead of flattening the file.
    If the JSON is malformed or has an unexpected shape, we fall back to indexing
    the raw text so the file is never silently dropped.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        records = data.get("contenido")
    except (json.JSONDecodeError, AttributeError):
        records = None

    if not isinstance(records, list):
        metadata = {"source": path.name, "format": "json"}
        return [Document(page_content=raw, metadata=metadata)]

    documents = []
    for record in records:
        documents.append(
            Document(
                page_content=_render_json_record(record),
                metadata=_json_record_metadata(record, path.name),
            )
        )
    return documents


def _render_json_record(record: dict) -> str:
    """Render one JSON error record into readable Spanish text for embedding."""
    lines = []
    if record.get("id"):
        lines.append(f"Código de error: {record['id']}")
    if record.get("titulo"):
        lines.append(record["titulo"])
    if record.get("mensaje_usuario"):
        lines.append(f"Mensaje al usuario: {record['mensaje_usuario']}")

    causas = record.get("causas_posibles", [])
    if causas:
        lines.append("Causas posibles:")
        for causa in causas:
            lines.append(f"- {causa}")

    solucion = record.get("solucion", [])
    if solucion:
        lines.append("Solución:")
        for paso in solucion:
            lines.append(f"- {paso}")

    if record.get("palabras_clave"):
        lines.append("Palabras clave: " + ", ".join(record["palabras_clave"]))

    return "\n".join(lines)


def _json_record_metadata(record: dict, source: str) -> dict:
    """Build Chroma-friendly metadata tags from a JSON record.

    ChromaDB only accepts scalar metadata values, so the ``palabras_clave`` list
    is joined into a single comma-separated string.
    """
    return {
        "source": source,
        "format": "json_record",
        "id": record.get("id", ""),
        "categoria": record.get("categoria", ""),
        "titulo": record.get("titulo", ""),
        "nivel_soporte": record.get("nivel_soporte", ""),
        "palabras_clave": ", ".join(record.get("palabras_clave", [])),
    }


# --------------------------------------------------------------------------- #
# Cleaning / normalization
# --------------------------------------------------------------------------- #
def clean_documents(documents: list[Document]) -> list[Document]:
    """Normalize each document's text in place (NFKC + tidy whitespace)."""
    for doc in documents:
        doc.page_content = _clean_text(doc.page_content)
    return documents


def _clean_text(text: str) -> str:
    """Fix unicode artifacts (e.g. the 'ﬁ' ligature) and collapse stray whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # unify newlines
    text = re.sub(r"[ \t]+", " ", text)        # collapse spaces/tabs within a line
    text = re.sub(r" *\n", "\n", text)         # drop trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse blank-line runs
    return text.strip()


# --------------------------------------------------------------------------- #
# Chunking (per file type)
# --------------------------------------------------------------------------- #
def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks, choosing the strategy by file format.

    Markdown is split structurally (by title) so each error stays coherent and
    keeps its tags; everything else (txt, pdf, and the already record-sized JSON
    Documents) goes through a recursive character splitter that only acts when a
    chunk exceeds the size limit. The same size splitter is reused for both so
    the size/overlap config lives in one place.
    """
    markdown_docs = []
    other_docs = []
    for doc in documents:
        if doc.metadata.get("format") == "md":
            markdown_docs.append(doc)
        else:
            other_docs.append(doc)

    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = _chunk_markdown(markdown_docs, size_splitter)
    chunks.extend(size_splitter.split_documents(other_docs))
    return chunks


def _chunk_markdown(
    documents: list[Document], size_splitter: RecursiveCharacterTextSplitter
) -> list[Document]:
    """Split markdown by its ``#`` title, tag each section, then cap it by size.

    In this corpus every ``.md`` file documents one error as an ``#`` title with
    ``##`` field sections (Código, Categoría, Causas, Solución, Palabras clave).
    Splitting on the title keeps the whole error together (causes and solution
    stay in one chunk). The header splitter drops the original metadata, so we
    merge ``source``/``format`` back in, add the title tag it produced, and lift
    the ``##`` fields into tags so markdown chunks are tagged like JSON records.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "titulo")],
        strip_headers=False,
    )

    chunks = []
    for doc in documents:
        sections = header_splitter.split_text(doc.page_content)
        for section in sections:
            section.metadata = {**doc.metadata, **section.metadata}
            _extract_markdown_tags(section)
        chunks.extend(size_splitter.split_documents(sections))
    return chunks


def _extract_markdown_tags(doc: Document) -> None:
    """Lift markdown ``##`` field sections into metadata tags, mirroring JSON.

    Reads the Código / Categoría / Palabras clave sections from the chunk body
    and stores them as ``id`` / ``categoria`` / ``palabras_clave`` tags (the
    ``titulo`` tag already comes from the ``#`` header).
    """
    codigo = _markdown_section(doc.page_content, "Código")
    if codigo:
        doc.metadata["id"] = codigo

    categoria = _markdown_section(doc.page_content, "Categoría")
    if categoria:
        doc.metadata["categoria"] = categoria

    palabras_clave = _markdown_section(doc.page_content, "Palabras clave")
    if palabras_clave:
        doc.metadata["palabras_clave"] = palabras_clave


def _markdown_section(text: str, header: str) -> str:
    """Return the text under a ``## <header>`` section, up to the next ``##``."""
    pattern = rf"##\s*{header}\s*\n(.*?)(?=\n##\s|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def main() -> None:
    """Load, clean, and chunk the corpus, then print a short per-file summary
    plus a sample of the metadata tags so chunking can be verified by eye."""
    documents = clean_documents(load_documents())
    chunks = chunk_documents(documents)

    counts = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counts[source] = counts.get(source, 0) + 1

    print(f"Loaded {len(documents)} documents, produced {len(chunks)} chunks:")
    for source in sorted(counts):
        print(f"  {source}: {counts[source]} chunks")

    print("\nSample chunk tags:")
    for chunk in chunks[:5]:
        tags = dict(chunk.metadata)
        tags.pop("source", None)
        print(f"  [{chunk.metadata['source']}] {len(chunk.page_content)} chars  tags={tags}")

    # Print the first 2 chunks from every file
    print("\nSample chunks:")
    seen = set()
    for chunk in chunks:
        source = chunk.metadata["source"]
        if (source, chunk.metadata.get("id")) not in seen:
            print(f"\n--- {source} ---")
            print(chunk.page_content + "...\n")
            print("===" * 10)
            seen.add((source, chunk.metadata.get("id")))
        if len(seen) >= 10:  # limit to first 10 files
            break


if __name__ == "__main__":
    main()
