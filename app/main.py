from fastapi import FastAPI
from app.api.router import router

app = FastAPI(
    title="Personal AI Assistant",
    description="GenAI-powered assistant for turning messy thoughts into structured actions",
    version="0.1.0",
)

app.include_router(router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
