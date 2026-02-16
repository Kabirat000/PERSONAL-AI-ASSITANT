"""
Ideas API routes.
RESTful endpoints for idea processing with RAG capabilities.
"""
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import (
    IdeaRequest,
    IdeaResponse,
    IdeaErrorResponse,
    MemoryItem,
    MemoryResponse,
    MemoryStatsResponse,
    ErrorResponse,
)
from app.services.idea_service import process_idea
from app.rag.retriever import get_all_memories, get_memory_stats, delete_idea
from app.core.exceptions import RAGException

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/",
    response_model=IdeaResponse,
    responses={
        200: {"description": "Successfully processed idea", "model": IdeaResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse},
    },
    summary="Process a new idea",
    description="Transform a raw thought into structured output using RAG",
)
def submit_idea(request: IdeaRequest) -> dict[str, Any]:
    """
    Process a raw thought and transform it into structured output.

    Uses RAG to find related past ideas and enhance the response.
    Optionally stores the idea in vector memory for future recall.
    """
    try:
        result = process_idea(
            raw_text=request.content,
            store_in_memory=request.store_in_memory,
        )

        # Check if result contains error
        if "error" in result:
            return result

        return result

    except RAGException as e:
        logger.error(f"RAG error processing idea: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "rag_error", "message": str(e)},
        )
    except Exception as e:
        logger.error(f"Unexpected error processing idea: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "message": "Failed to process idea"},
        )


@router.get(
    "/memory",
    response_model=MemoryResponse,
    summary="Get all stored memories",
    description="Retrieve all ideas stored in vector memory (debug endpoint)",
    tags=["Debug"],
)
def read_memory() -> dict[str, Any]:
    """
    Retrieve all stored ideas from vector memory.

    This is primarily a debug/admin endpoint for inspecting stored data.
    """
    try:
        memories = get_all_memories()
        return {
            "count": len(memories),
            "memories": memories,
        }
    except RAGException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "vector_store_error", "message": str(e)},
        )


@router.get(
    "/memory/stats",
    response_model=MemoryStatsResponse,
    summary="Get memory statistics",
    description="Get statistics about the vector memory store",
    tags=["Debug"],
)
def get_stats() -> dict[str, Any]:
    """
    Get statistics about the vector memory store.

    Returns total count, embedding provider info, and health status.
    """
    try:
        return get_memory_stats()
    except RAGException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "vector_store_error", "message": str(e)},
        )


@router.delete(
    "/memory/{doc_id}",
    response_model=dict,
    summary="Delete a memory",
    description="Delete a specific idea from vector memory",
    tags=["Debug"],
)
def remove_memory(doc_id: str) -> dict[str, Any]:
    """
    Delete a specific idea from vector memory by its ID.

    Args:
        doc_id: The document ID to delete
    """
    try:
        success = delete_idea(doc_id)
        return {"success": success, "deleted_id": doc_id}
    except RAGException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "vector_store_error", "message": str(e)},
        )
