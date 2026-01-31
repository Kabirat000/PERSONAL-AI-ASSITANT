from fastapi import APIRouter
from pydantic import BaseModel
from app.services.task_service import extract_tasks

router = APIRouter()


class TaskExtractRequest(BaseModel):
    content: str


@router.post("/extract")
def extract_tasks_endpoint(request: TaskExtractRequest):
    tasks = extract_tasks(request.content)
    return {
        "count": len(tasks),
        "tasks": tasks
    }
