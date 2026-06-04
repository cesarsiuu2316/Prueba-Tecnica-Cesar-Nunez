"""Vector store for the corpus.

Stores document chunks and their embeddings in a persistent ChromaDB collection
configured for cosine similarity. IDs are derived from the chunk content, and
``upsert`` is used so re-running ingestion never creates duplicates.
"""

import hashlib
import os
import chromadb
import numpy as np
from dotenv import load_dotenv
from langchain_core.documents import Document
from data_ingest import chunk_documents, clean_documents, load_documents
from embedding_docs import EmbeddingManager

load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "docs/chroma-db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "ChromaDB_collection_for_corpus")


class VectorStore:
    """Persistent ChromaDB store for chunk embeddings (cosine distance)."""

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        persist_directory: str = CHROMA_DIR,
    ):
        """Open (or create) the persistent collection.

        Args:
            collection_name: name of the ChromaDB collection.
            persist_directory: folder where ChromaDB persists its data.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_chromadb()

    def _initialize_chromadb(self):
        """Create the persistent client and the cosine collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                # cosine over the normalized embeddings produced by EmbeddingManager
                configuration={"hnsw": {"space": "cosine"}},
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise

    def add_documents(self, documents: list[Document], embeddings: np.ndarray) -> None:
        """Upsert chunks and their embeddings into the collection.

        Args:
            documents: the chunks to store.
            embeddings: one embedding row per chunk (same order as documents).
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids = []
        metadatas = []
        texts = []
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            ids.append(self._chunk_id(doc))
            metadatas.append(dict(doc.metadata))
            texts.append(doc.page_content)
            vectors.append(embedding.tolist())

        # upsert (not add) so identical IDs overwrite instead of duplicating
        try: 
            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=texts,
            )
        except Exception as e:
            print(f"Error upserting documents into ChromaDB: {e}")
            raise

    def count(self) -> int:
        """Return how many chunks are currently stored."""
        return self.collection.count()

    @staticmethod
    def _chunk_id(doc: Document) -> str:
        """Deterministic ID from source + content (makes re-ingestion idempotent)."""
        source = doc.metadata.get("source", "")
        raw = f"{source}|{doc.page_content}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def main() -> None:
    """Build the index end to end: load -> clean -> chunk -> embed -> store."""

    chunks = chunk_documents(clean_documents(load_documents()))

    texts = []
    for chunk in chunks:
        texts.append(chunk.page_content)

    embeddings = EmbeddingManager().embed_documents(texts)

    store = VectorStore()
    store.add_documents(chunks, embeddings)
    print(f"Stored {len(chunks)} chunks. Collection holds {store.count()} chunks.")


if __name__ == "__main__":
    main()
