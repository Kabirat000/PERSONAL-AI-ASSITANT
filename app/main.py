"""
Main FastAPI application entry point.
Configures the application with all routes, middleware, and lifecycle events.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import router
from app.api.schemas import HealthResponse
from app.core.config import get_settings
from app.core.exceptions import RAGException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logger.info("Starting Personal AI Assistant API...")
    settings = get_settings()
    logger.info(f"Using embedding provider: Voyage AI ({settings.voyage_embedding_model})")
    logger.info(f"Using LLM model: Groq ({settings.groq_model})")
    logger.info(f"Qdrant host: {settings.qdrant_host}:{settings.qdrant_port}")

    yield

    # Shutdown
    logger.info("Shutting down Personal AI Assistant API...")


# Initialize the FastAPI app with metadata
app = FastAPI(
    title="Personal AI Assistant",
    description="""
## AI-Powered Personal Assistant

Transform messy thoughts into clear, actionable intelligence using advanced NLP and RAG.

### Features
- **Idea Processing**: Convert raw thoughts into structured notes with themes and tasks
- **Task Extraction**: Identify actionable items from unstructured text
- **Semantic Memory**: Store and recall related ideas using vector similarity search
- **Multi-Provider Embeddings**: Support for OpenAI and Voyage AI embedding models

### API Endpoints
- `/ideas/` - Process and store ideas with RAG enhancement
- `/tasks/extract` - Extract actionable tasks from text
- `/health` - Check service health status
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler for RAG exceptions
@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """Handle RAG-related exceptions globally."""
    logger.error(f"RAG Exception: {exc.message}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "rag_error",
            "message": exc.message,
            "details": exc.details,
        },
    )


# Include routers
app.include_router(router)


# Root endpoint
@app.get(
    "/",
    summary="API Root",
    description="Welcome message and API information",
)
def read_root() -> dict[str, Any]:
    """Return API welcome message and version info."""
    return {
        "message": "Welcome to the Personal AI Assistant API!",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of all services",
)
def health_check() -> dict[str, Any]:
    """
    Check health status of all dependent services.

    Returns status of:
    - API server
    - Qdrant vector store
    - Embedding service
    """
    settings = get_settings()
    services = {}

    # Check Qdrant
    try:
        from app.rag.retriever import get_vector_store
        vector_store = get_vector_store()
        services["qdrant"] = vector_store.health_check()
    except Exception as e:
        services["qdrant"] = {"status": "unhealthy", "error": str(e)}

    # Embedding service info
    services["embedding"] = {
        "status": "healthy",
        "provider": "voyage",
        "model": settings.voyage_embedding_model,
        "dimension": settings.active_embedding_dim,
    }

    # LLM info
    services["llm"] = {
        "status": "healthy",
        "provider": "groq",
        "model": settings.groq_model,
    }

    # Determine overall status
    overall_status = "ok"
    if services["qdrant"].get("status") == "unhealthy":
        overall_status = "degraded"

    return {
        "status": overall_status,
        "version": "1.0.0",
        "services": services,
    }
