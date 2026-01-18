from typing import List
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantVectorStore:
    def __init__(self, collection_name: str, embedding_dim: int):
        self.collection_name = collection_name
        self.client = QdrantClient(host="qdrant", port=6333)

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    def add(self, embedding: List[float], text: str) -> None:
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": text},
        )
        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def search(self, embedding: List[float], top_k: int = 3) -> List[str]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
        )
        return [hit.payload["text"] for hit in results]

    # 🔍 DEBUG
    def get_all(self, limit: int = 100) -> List[str]:
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
        )
        return [p.payload["text"] for p in records]
