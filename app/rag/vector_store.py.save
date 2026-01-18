from typing import List
import faiss
import numpy as np


class FaissVectorStore:
    """
    In-memory FAISS vector store.
    This is intentionally simple and non-persistent.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts: List[str] = []

    def add(self, embedding: List[float], text: str) -> None:
        vector = np.array([embedding]).astype("float32")
        self.index.add(vector)
        self.texts.append(text)

    def search(self, embedding: List[float], top_k: int = 3) -> List[str]:
        if self.index.ntotal == 0:
            return []

        vector = np.array([embedding]).astype("float32")
        _, indices = self.index.search(vector, top_k)

        return [
            self.texts[i]
            for i in indices[0]
            if i < len(self.texts)
        ]

