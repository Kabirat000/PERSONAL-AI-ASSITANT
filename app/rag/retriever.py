"""
RAG Retriever module.
Provides high-level functions for storing and retrieving ideas using semantic search.
"""
import logging
from datetime import datetime, timezone
from typing import Any

from app.services.embedding_service import get_embedding_service
from app.rag.qdrant_store import QdrantVectorStore
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Initialize services
_vector_store: QdrantVectorStore | None = None
_embedding_service = None


def get_vector_store() -> QdrantVectorStore:
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        _vector_store = QdrantVectorStore(
            collection_name=settings.qdrant_collection_name,
            embedding_dim=settings.active_embedding_dim,
        )
    return _vector_store


def store_idea(
    text: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Store a new idea in the vector database.

    Args:
        text: The idea text to store
        metadata: Optional additional metadata

    Returns:
        The generated document ID
    """
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()

    # Generate embedding using document mode for storage
    embedding = embedding_service.embed_document(text)

    # Add metadata
    full_metadata = {
        "type": "idea",
        "stored_at": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
    }

    return vector_store.add(embedding, text, full_metadata)


def retrieve_similar_ideas(
    text: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[str]:
    """
    Retrieve ideas similar to the given text.

    Args:
        text: Query text to find similar ideas
        top_k: Number of results to return
        score_threshold: Minimum similarity score

    Returns:
        List of similar idea texts
    """
    settings = get_settings()
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()

    top_k = top_k or settings.rag_top_k
    score_threshold = score_threshold or settings.rag_score_threshold

    # Generate embedding using query mode for retrieval
    embedding = embedding_service.embed_query(text)

    return vector_store.search_texts(embedding, top_k, score_threshold)


def retrieve_similar_ideas_with_scores(
    text: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve ideas with similarity scores and metadata.

    Args:
        text: Query text to find similar ideas
        top_k: Number of results to return
        score_threshold: Minimum similarity score

    Returns:
        List of results with text, score, and metadata
    """
    settings = get_settings()
    embedding_service = get_embedding_service()
    vector_store = get_vector_store()

    top_k = top_k or settings.rag_top_k
    score_threshold = score_threshold or settings.rag_score_threshold

    embedding = embedding_service.embed_query(text)
    return vector_store.search(embedding, top_k, score_threshold)


def get_all_memories() -> list[dict[str, Any]]:
    """
    Get all stored memories/ideas (for debugging).

    Returns:
        List of all stored documents with metadata
    """
    vector_store = get_vector_store()
    return vector_store.get_all()


def delete_idea(doc_id: str) -> bool:
    """
    Delete an idea by its ID.

    Args:
        doc_id: The document ID to delete

    Returns:
        True if deletion was successful
    """
    vector_store = get_vector_store()
    return vector_store.delete(doc_id)


def get_memory_stats() -> dict[str, Any]:
    """
    Get statistics about stored memories.

    Returns:
        Dictionary with count and health status
    """
    vector_store = get_vector_store()
    embedding_service = get_embedding_service()

    return {
        "total_ideas": vector_store.count(),
        "embedding_provider": embedding_service.provider_name,
        "embedding_dimension": embedding_service.dimension,
        "health": vector_store.health_check(),
    }
