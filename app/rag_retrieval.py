"""Retrieval for RAG API.

Embeds a user question, runs cosine search over the ChromaDB collection, keeps
only the chunks above a similarity threshold, and assembles a Spanish grounding
prompt. When nothing is relevant enough it reports that no context was found, so
the caller (the external n8n LLM node) never fabricates an answer.
"""

import os
from dotenv import load_dotenv
from embedding_docs import EmbeddingManager
from vector_store import VectorStore

load_dotenv()
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))

# Sent to the external LLM node; enforces grounding and Spanish answers.
SYSTEM_INSTRUCTION = (
    "Eres un asistente de soporte técnico de UniLink. Responde la pregunta "
    "utilizando únicamente la información del CONTEXTO. Si el contexto no "
    "contiene la respuesta, indica explícitamente que no dispones de esa "
    "información en la documentación. Responde siempre en español."
)


class RAGRetriever:
    """Retrieves grounded context for a question and builds a Spanish prompt."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        """Wire the embedding model and the vector store together.

        Args:
            top_k: how many candidate chunks to pull from the store.
            similarity_threshold: minimum cosine similarity to keep a chunk.
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.embedder = embedding_manager
        self.store = vector_store

    def retrieve(self, question: str) -> dict:
        """Return grounded context and a Spanish prompt for a question.

        The result always contains ``sufficient_context``: when it is False the
        caller should reply that the documentation has no answer.
        """
        question = question.strip()
        if not question:
            return self._result(question, [])

        contexts = self._search(question)
        return self._result(question, contexts)

    def _search(self, question: str) -> list[dict]:
        """Embed the question, query the store, and keep chunks above threshold."""
        embedding = self.embedder.embed_query(question)
        results = self.store.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        contexts = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1 - distance  # cosine distance -> similarity
            if similarity >= self.similarity_threshold:
                contexts.append(
                    {
                        "text": text,
                        "source": metadata.get("source", ""),
                        "similarity": round(similarity, 3),
                    }
                )
        return contexts

    def _result(self, question: str, contexts: list[dict]) -> dict:
        """Shape the response and attach the assembled grounding prompt."""
        return {
            "question": question,
            "sufficient_context": bool(contexts),
            "contexts": contexts,
            "prompt": self._build_prompt(question, contexts),
        }

    def _build_prompt(self, question: str, contexts: list[dict]) -> str:
        """Assemble the Spanish grounding prompt for the external LLM node."""
        if not contexts:
            context_block = "(No se encontró contexto relevante en la documentación.)"
        else:
            blocks = []
            for i, ctx in enumerate(contexts, start=1):
                blocks.append(f"[{i}] (Fuente: {ctx['source']})\n{ctx['text']}")
            context_block = "\n\n".join(blocks)

        return (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"CONTEXTO:\n{context_block}\n\n"
            f"PREGUNTA: {question}\n\n"
            f"RESPUESTA:"
        )


def main() -> None:
    """Quick check: a relevant question and an off-topic one."""
    retriever = RAGRetriever(embedding_manager=EmbeddingManager(), vector_store=VectorStore())
    questions = [
        "Contraseña y credenciales de login incorrectas?",
        "¿Cuál es la capital de Francia?",
        "Error material con código duplicado",
        "el formato de archivo no es compatible"
    ]
    for question in questions:
        result = retriever.retrieve(question)
        print(f"\nPregunta: {question}")
        print(f"  sufficient_context: {result['sufficient_context']}")
        for ctx in result["contexts"]:
            print(f"  - {ctx['source']} (sim={ctx['similarity']})")
        print("Print the whole result assembled prompt: ")
        print(result["prompt"])
        print("=======================================================")

if __name__ == "__main__":
    main()
