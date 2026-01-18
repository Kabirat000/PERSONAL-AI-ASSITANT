from app.services.embedding_service import embed_text, EMBEDDING_DIM
from app.rag.qdrant_store import QdrantVectorStore

vector_store = QdrantVectorStore(
    collection_name="ideas",
    embedding_dim=EMBEDDING_DIM,
)


def store_idea(text: str) -> None:
    embedding = embed_text(text)
    vector_store.add(embedding, text)


def retrieve_similar_ideas(text: str, top_k: int = 3) -> list[str]:
    embedding = embed_text(text)
    return vector_store.search(embedding, top_k)


# 🔍 DEBUG — QDRANT
def get_all_memories() -> list[str]:
    return vector_store.get_all()


