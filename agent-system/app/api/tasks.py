from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from app.agents.memory.user_feedback_memory import UserFeedbackMemory
import uuid
from datetime import datetime, timezone
import structlog
from app.agents.memory.agent_performance_memory import AgentPerformanceMemory
from app.agents.memory.agent_preference_memory import AgentPreferenceMemory
from app.agents.memory.tool_success_memory import ToolSuccessMemory
from app.db.session import get_db
from app.models.task import Task, Step, TaskStatus
from app.orchestrator.loop_v3 import execute_task_v3 as execute_task

logger = structlog.get_logger()
router = APIRouter()

agent_performance_memory = AgentPerformanceMemory()
agent_preference_memory = AgentPreferenceMemory()
tool_success_memory = ToolSuccessMemory()
feedback_memory = UserFeedbackMemory()


class FeedbackRequest(BaseModel):
    query: str
    feedback: str


class TaskCreate(BaseModel):
    prompt: str
    session_id: Optional[str] = None


class StepResponse(BaseModel):
    id: str
    step_number: int
    instruction: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int

    class Config:
        from_attributes = True


class TaskResponse(BaseModel):
    task_id: str
    user_input: str
    status: str
    created_at: str
    session_id: Optional[str] = None
    steps: List[StepResponse] = []

    class Config:
        from_attributes = True


@router.post("/tasks", response_model=TaskResponse)
async def create_task(
    task_data: TaskCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a new task and start execution"""
    task_id = str(uuid.uuid4())

    task = Task(
        id=task_id,
        user_input=task_data.prompt,
        status=TaskStatus.PENDING,
        session_id=task_data.session_id,
    )

    db.add(task)
    db.commit()
    db.refresh(task)

    logger.info("task_created", task_id=task_id, user_input=task_data.prompt)

    # Execute task in background
    background_tasks.add_task(execute_task, task_id)

    return TaskResponse(
        task_id=task.id,
        user_input=task_data.prompt,
        status=task.status,
        created_at=task.created_at.isoformat(),
        session_id=task.session_id,
        steps=[],
    )


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, db: Session = Depends(get_db)):
    """Get task status and results"""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    steps = [
        StepResponse(
            id=step.id,
            step_number=step.step_number,
            instruction=step.instruction,
            status=step.status,
            result=step.result,
            error=step.error,
            retry_count=step.retry_count,
        )
        for step in sorted(task.steps, key=lambda s: s.step_number)
    ]

    return TaskResponse(
        task_id=task.id,
        user_input=task.user_input,
        session_id=task.session_id,
        status=task.status,
        created_at=task.created_at.isoformat(),
        steps=steps,
    )


@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(limit: int = 10, offset: int = 0, db: Session = Depends(get_db)):
    """List all tasks"""
    tasks = (
        db.query(Task)
        .order_by(Task.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        TaskResponse(
            task_id=task.id,
            user_input=task.user_input,
            status=task.status,
            created_at=task.created_at.isoformat(),
            session_id=task.session_id,
            steps=[
                StepResponse(
                    id=step.id,
                    step_number=step.step_number,
                    instruction=step.instruction,
                    status=step.status,
                    result=step.result,
                    error=step.error,
                    retry_count=step.retry_count,
                )
                for step in sorted(task.steps, key=lambda s: s.step_number)
            ],
        )
        for task in tasks
    ]


@router.get("/files")
async def list_files():
    """List all files in the shared workspace"""
    from app.utils.file_manager import FileManager

    fm = FileManager()
    files = fm.list_files()
    return {"files": sorted(files), "count": len(files)}


@router.get("/files/{filename}")
async def get_file(filename: str):
    from app.utils.file_manager import FileManager
    from fastapi.responses import PlainTextResponse

    fm = FileManager()
    content = fm.read_file(filename)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")
    return PlainTextResponse(content=content)


@router.get("/analytics")
async def get_analytics():
    """Get analytics from cost tracker files"""
    import json
    from pathlib import Path
    from app.core.config import settings

    costs_dir = Path(settings.COSTS_DIR)
    files = list(costs_dir.glob("task_*.json"))

    if not files:
        return {"total_tasks": 0, "tasks": []}

    tasks = []
    for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
        try:
            with open(f) as fp:
                data = json.load(fp)
            tasks.append(data)
        except:
            continue

    total = len(tasks)
    successful = sum(1 for t in tasks if t.get("success"))
    failed = total - successful

    return {
        "total_tasks": total,
        "successful": successful,
        "failed": failed,
        "success_rate": round(successful / total * 100, 1) if total > 0 else 0,
        "avg_duration": round(sum(t.get("duration_sec", 0) for t in tasks) / total, 1),
        "avg_llm_calls": round(
            sum(t.get("total_llm_calls", 0) for t in tasks) / total, 1
        ),
        "avg_cost_usd": round(
            sum(t.get("estimated_cost_usd", 0) for t in tasks) / total, 6
        ),
        "total_cost_usd": round(sum(t.get("estimated_cost_usd", 0) for t in tasks), 4),
        "avg_retries": round(sum(t.get("total_retries", 0) for t in tasks) / total, 1),
        "tasks": tasks[:20],
    }


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, db: Session = Depends(get_db)):
    """Cancel a running task"""
    from app.utils.websocket_manager import cancellation_store

    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status not in ["PENDING", "RUNNING"]:
        raise HTTPException(
            status_code=400, detail=f"Task is {task.status} — cannot cancel"
        )

    # Signal cancellation
    cancellation_store.cancel(task_id)

    # Update DB
    task.status = TaskStatus.FAILED
    task.error_message = "Cancelled by user"
    task.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    # Notify WebSocket
    from app.utils.websocket_manager import ws_manager

    await ws_manager.emit(task_id, {"phase": "cancelled", "status": "cancelled"})

    logger.info("task_cancelled", task_id=task_id)
    return {"task_id": task_id, "status": "cancelled"}


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    feedback_memory.record_feedback(
        query=request.query,
        feedback=request.feedback,
    )

    return {"success": True}
@router.get("/feedback/stats")
async def feedback_stats():
    return feedback_memory.get_feedback_stats()

@router.get("/memory/stats")
async def memory_stats():

    return {
        "feedback_entries": len(
            feedback_memory.feedback
        ),

        "agent_performance_entries": len(
            agent_performance_memory.all()
        ),

        "agent_preference_entries": len(
            agent_preference_memory.preferences
        ),

        "tool_success_entries": len(
            tool_success_memory.memory
        ),
    }
@router.get("/agents/performance")
async def agent_performance():
    return agent_performance_memory.all()

@router.get("/tools/performance")
async def tool_performance():
    return tool_success_memory.memory

@router.get("/agents/preferences")
async def agent_preferences():
    return agent_preference_memory.preferences

@router.get("/feedback/history")
async def feedback_history():
    return feedback_memory.feedback[-50:]