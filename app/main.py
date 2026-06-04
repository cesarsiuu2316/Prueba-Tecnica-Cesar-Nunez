"""Entry point: ingest the corpus and persist it to the vector store.

Runs the full pipeline once - load -> clean -> chunk -> embed -> store - so the
retrieval API has a populated ChromaDB collection to search.

Usage:
    python app/main.py
"""
import os
# Suppress HuggingFace Hub warnings about symlinks and progress bars 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

from data_ingest import chunk_documents, clean_documents, load_documents
from embedding_docs import EmbeddingManager
from vector_store import VectorStore


def ingest() -> int:
    """Load, clean, chunk, embed, and store the corpus. Returns the chunk count."""
    documents = load_documents()
    documents = clean_documents(documents)
    chunks = chunk_documents(documents)

    texts = []
    for chunk in chunks:
        texts.append(chunk.page_content)

    embeddings = EmbeddingManager().embed_documents(texts)

    store = VectorStore()
    store.add_documents(chunks, embeddings)
    return store.count()


def main() -> None:
    """Run ingestion and report how many chunks ended up in the vector store."""
    print("Ingesting corpus...")
    total = ingest()
    print(f"Done. Vector store holds {total} chunks.")


if __name__ == "__main__":
    main()
