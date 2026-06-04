"""Embedding management for the corpus.

Wraps a local SentenceTransformer model (multilingual-e5-small by default) to
turn document chunks and user queries into normalized vectors for cosine search.
The e5 model family expects "passage:" / "query:" prefixes, so documents and
queries are embedded through separate methods.
"""

import os
# Suppress HuggingFace Hub warnings about symlinks and progress bars 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

class EmbeddingManager:
    """Creates normalized embeddings for documents and queries."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Load the SentenceTransformer model (GPU if available, else CPU).

        Args:
            model_name: HuggingFace model id to load.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the SentenceTransformer model on the appropriate device."""
        try: 
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            print(f"Error loading model '{self.model_name}' on {self.device}: {e}")
            raise

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed corpus chunks. e5 expects the 'passage:' prefix."""
        prefixed = []
        for text in texts:
            prefixed.append(f"passage: {text}")
        return self._encode(prefixed)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single user query. e5 expects the 'query:' prefix."""
        return self._encode([f"query: {text}"])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into L2-normalized vectors (for cosine similarity)."""
        return self.model.encode(texts, normalize_embeddings=True)


def main() -> None:
    """Quick check: embed a couple of Spanish samples and report shape + norm."""
    manager = EmbeddingManager()
    vectors = manager.embed_documents(
        [
            "El sistema devuelve error 502.",
            "No puedo conectar con la base de datos.",
        ]
    )
    print(f"Model: {manager.model_name} on {manager.device}")
    print(f"Embeddings shape: {vectors.shape}")
    print(f"First vector norm: {np.linalg.norm(vectors[0]):.3f}")


if __name__ == "__main__":
    main()
