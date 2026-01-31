from fastapi import FastAPI
from app.api.router import router

# Initialize the FastAPI app with metadata
app = FastAPI(
    title="Personal AI Assistant",
    description="GenAI-powered assistant for turning messy thoughts into structured actions",
    version="0.1.0",
)

# Include your routers to define API routes
app.include_router(router)

# Root route to return a basic message when accessing /
@app.get("/")
def read_root():
    return {"message": "Welcome to the Personal AI Assistant API!"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}
