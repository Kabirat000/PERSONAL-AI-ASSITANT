"""
Custom exceptions for the RAG application.
Provides structured error handling with proper HTTP status codes.
"""
from fastapi import HTTPException, status


class RAGException(Exception):
    """Base exception for RAG-related errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class LLMError(RAGException):
    """Raised when LLM inference fails."""
    pass


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid."""
    pass


# ─────────────────────────────────────────────────────────────────
# HTTP Exception Factories
# ─────────────────────────────────────────────────────────────────

def raise_embedding_error(message: str) -> HTTPException:
    """Create HTTP exception for embedding errors."""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": "embedding_error", "message": message},
    )


def raise_llm_error(message: str) -> HTTPException:
    """Create HTTP exception for LLM errors."""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": "llm_error", "message": message},
    )


def raise_vector_store_error(message: str) -> HTTPException:
    """Create HTTP exception for vector store errors."""
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"error": "vector_store_error", "message": message},
    )


def raise_validation_error(message: str) -> HTTPException:
    """Create HTTP exception for validation errors."""
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"error": "validation_error", "message": message},
    )
