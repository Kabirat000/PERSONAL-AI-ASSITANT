"""
Pydantic schemas for API requests and responses.
Provides type-safe, documented models for the RESTful API.
"""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────
# Request Schemas
# ─────────────────────────────────────────────────────────────────

class IdeaRequest(BaseModel):
    """Request body for submitting a new idea."""
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The raw thought or idea to process",
        examples=["I need to finish my thesis by next week and also call mom"],
    )
    store_in_memory: bool = Field(
        default=True,
        description="Whether to store this idea for future recall",
    )


class TaskExtractRequest(BaseModel):
    """Request body for extracting tasks from text."""
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The text to extract tasks from",
        examples=["Need to buy groceries, call the dentist, and finish the report by Friday"],
    )


# ─────────────────────────────────────────────────────────────────
# Response Schemas
# ─────────────────────────────────────────────────────────────────

class TaskItem(BaseModel):
    """A single extracted task."""
    task: str = Field(..., description="The actionable task description")
    priority: Literal["high", "medium", "low"] = Field(
        ...,
        description="Task priority level",
    )


class IdeaResponse(BaseModel):
    """Response from idea processing."""
    clean_note: str = Field(..., description="The refined, structured version of the thought")
    themes: list[str] = Field(default_factory=list, description="Identified themes or topics")
    suggested_tasks: list[TaskItem] = Field(
        default_factory=list,
        description="Actionable tasks extracted from the idea",
    )
    context_used: bool = Field(
        default=False,
        description="Whether related past ideas were used for context",
    )
    related_ideas_count: int = Field(
        default=0,
        description="Number of related ideas found",
    )


class IdeaErrorResponse(BaseModel):
    """Error response from idea processing."""
    error: str
    raw_output: str | None = None
    context_used: bool = False


class TaskExtractResponse(BaseModel):
    """Response from task extraction."""
    count: int = Field(..., description="Number of tasks extracted")
    tasks: list[TaskItem] = Field(..., description="List of extracted tasks")


class MemoryItem(BaseModel):
    """A single stored memory/idea."""
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Response from memory retrieval."""
    count: int = Field(..., description="Total number of stored memories")
    memories: list[MemoryItem] = Field(..., description="List of stored memories")


class MemoryStatsResponse(BaseModel):
    """Response from memory stats endpoint."""
    total_ideas: int
    embedding_provider: str
    embedding_dimension: int
    health: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["ok", "degraded", "unhealthy"]
    version: str
    services: dict[str, dict[str, Any]]


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")
