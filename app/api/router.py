from fastapi import APIRouter
from app.api.routes import ideas, tasks

router = APIRouter()

router.include_router(ideas.router, prefix="/ideas", tags=["Ideas"])
router.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
