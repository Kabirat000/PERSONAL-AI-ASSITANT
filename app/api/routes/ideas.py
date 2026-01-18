from fastapi import APIRouter
from pydantic import BaseModel

from app.services.idea_service import process_idea
from app.rag.retriever import get_all_memories

router = APIRouter()


class IdeaRequest(BaseModel):
    content: str


@router.post("/")
def submit_idea(request: IdeaRequest):
    return process_idea(request.content)


# 🔍 DEBUG ONLY — view memory contents
@router.get("/memory", tags=["Debug"])
def read_memory():
    memories = get_all_memories()
    return {
        "count": len(memories),
        "memories": memories,
    }
