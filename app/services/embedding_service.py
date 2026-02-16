"""
Voyage AI embedding service.
Provides text embeddings using Voyage AI's API for RAG retrieval.
"""
import logging
from typing import Literal

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"


class EmbeddingService:
    """
    Embedding service using Voyage AI.
    Supports query/document input types for optimized retrieval.
    """

    def __init__(self):
        settings = get_settings()
        if not settings.voyage_api_key:
            raise EmbeddingError("Voyage API key is required but not configured")
        self.api_key = settings.voyage_api_key
        self.model = settings.voyage_embedding_model
        self._dimension = settings.voyage_embedding_dim

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _call_api(
        self,
        texts: str | list[str],
        input_type: Literal["query", "document"] | None = None,
    ) -> list[list[float]]:
        """Call Voyage AI embedding API."""
        try:
            with httpx.Client(timeout=30.0) as client:
                payload = {
                    "input": texts,
                    "model": self.model,
                }
                if input_type:
                    payload["input_type"] = input_type

                response = client.post(
                    VOYAGE_API_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(f"Voyage AI API error ({e.response.status_code}): {error_body}")
            raise EmbeddingError(f"Voyage AI API error: {error_body}")
        except httpx.HTTPError as e:
            logger.error(f"Voyage AI request failed: {e}")
            raise EmbeddingError(f"Voyage AI request failed: {str(e)}")

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a search query."""
        results = self._call_api(text, input_type="query")
        return results[0]

    def embed_document(self, text: str) -> list[float]:
        """Generate embedding for a document to be stored."""
        results = self._call_api(text, input_type="document")
        return results[0]

    def embed_batch(
        self,
        texts: list[str],
        input_type: Literal["query", "document"] = "document",
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return self._call_api(texts, input_type=input_type)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "voyage"


# ─────────────────────────────────────────────────────────────────
# Module-level singleton and convenience functions
# ─────────────────────────────────────────────────────────────────

_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service


def embed_text(text: str) -> list[float]:
    """Embed text as a document (convenience function)."""
    return get_embedding_service().embed_document(text)


def embed_query(text: str) -> list[float]:
    """Embed text as a query (convenience function)."""
    return get_embedding_service().embed_query(text)


# Export embedding dimension
EMBEDDING_DIM = get_settings().active_embedding_dim
