"""
Tasks API routes.
RESTful endpoints for task extraction from text.
"""
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import TaskExtractRequest, TaskExtractResponse, ErrorResponse
from app.services.task_service import extract_tasks
from app.core.exceptions import RAGException

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/extract",
    response_model=TaskExtractResponse,
    responses={
        200: {"description": "Successfully extracted tasks", "model": TaskExtractResponse},
        422: {"description": "Validation error", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse},
    },
    summary="Extract tasks from text",
    description="Identify and extract actionable tasks from raw text input",
)
def extract_tasks_endpoint(request: TaskExtractRequest) -> dict[str, Any]:
    """
    Extract actionable tasks from raw text input.

    Uses LLM to identify specific, doable tasks and assign priority levels.
    """
    try:
        tasks = extract_tasks(request.content)
        return {
            "count": len(tasks),
            "tasks": tasks,
        }
    except RAGException as e:
        logger.error(f"Task extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "llm_error", "message": str(e)},
        )
    except Exception as e:
        logger.error(f"Unexpected error in task extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "internal_error", "message": "Failed to extract tasks"},
        )
