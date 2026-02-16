"""
Qdrant Vector Store implementation.
Handles storage and retrieval of embeddings with metadata support.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Production-ready Qdrant vector store with:
    - Safe collection initialization (no data loss on restart)
    - Configurable connection settings
    - Metadata support for documents
    - Score-based filtering
    """

    def __init__(
        self,
        collection_name: str | None = None,
        embedding_dim: int | None = None,
    ):
        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.embedding_dim = embedding_dim or settings.active_embedding_dim

        # Initialize client with configurable settings
        # Use HTTPS only when API key is provided (for Qdrant Cloud)
        use_https = bool(settings.qdrant_api_key)
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            https=use_https,
        )

        # Safely initialize collection (preserve existing data)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection only if it doesn't exist (preserves data on restart)."""
        try:
            collections = self.client.get_collections()
            existing_names = [c.name for c in collections.collections]

            if self.collection_name not in existing_names:
                logger.info(f"Creating new collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.embedding_dim,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
            else:
                logger.info(f"Using existing collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise VectorStoreError(f"Failed to initialize Qdrant collection: {str(e)}")

    def add(
        self,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Add a single document with its embedding to the store.

        Args:
            embedding: The vector embedding
            text: The original text content
            metadata: Optional additional metadata

        Returns:
            The generated document ID
        """
        doc_id = str(uuid.uuid4())
        payload = {
            "text": text,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )
            return doc_id
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise VectorStoreError(f"Failed to store document: {str(e)}")

    def add_batch(
        self,
        embeddings: list[list[float]],
        texts: list[str],
        metadata_list: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """
        Add multiple documents in a single batch operation.

        Args:
            embeddings: List of vector embeddings
            texts: List of original text content
            metadata_list: Optional list of metadata dicts

        Returns:
            List of generated document IDs
        """
        if len(embeddings) != len(texts):
            raise VectorStoreError("Embeddings and texts must have same length")

        metadata_list = metadata_list or [{} for _ in texts]
        doc_ids = []
        points = []

        for embedding, text, metadata in zip(embeddings, texts, metadata_list):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            points.append(
                qdrant_models.PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        **metadata,
                    },
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            return doc_ids
        except Exception as e:
            logger.error(f"Failed to add batch: {e}")
            raise VectorStoreError(f"Failed to store documents: {str(e)}")

    def search(
        self,
        embedding: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
        filter_conditions: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_conditions: Optional Qdrant filter conditions

        Returns:
            List of results with text, score, and metadata
        """
        settings = get_settings()
        top_k = top_k or settings.rag_top_k
        score_threshold = score_threshold or settings.rag_score_threshold

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_models.Filter(**filter_conditions) if filter_conditions else None,
            )
            results = response.points

            return [
                {
                    "id": str(hit.id),
                    "text": hit.payload.get("text", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {str(e)}")

    def search_texts(
        self,
        embedding: list[float],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[str]:
        """
        Search and return only text content (backward compatible).

        Args:
            embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of matching text strings
        """
        results = self.search(embedding, top_k, score_threshold)
        return [r["text"] for r in results]

    def get_all(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Retrieve all documents (for debugging/admin purposes).

        Args:
            limit: Maximum number of documents to return

        Returns:
            List of all stored documents with metadata
        """
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
            )
            return [
                {
                    "id": str(p.id),
                    "text": p.payload.get("text", ""),
                    "metadata": {k: v for k, v in p.payload.items() if k != "text"},
                }
                for p in records
            ]
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            raise VectorStoreError(f"Failed to retrieve documents: {str(e)}")

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID.

        Args:
            doc_id: The document ID to delete

        Returns:
            True if deletion was successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(points=[doc_id]),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise VectorStoreError(f"Failed to delete document: {str(e)}")

    def count(self) -> int:
        """Get the total number of documents in the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0

    def health_check(self) -> dict[str, Any]:
        """Check Qdrant connection health."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "status": "healthy",
                "collection": self.collection_name,
                "points_count": getattr(info, "points_count", 0),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
