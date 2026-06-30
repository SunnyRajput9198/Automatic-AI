import asyncio
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from pydantic import BaseModel
from typing import List, Optional
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
import structlog

from app.agents.memory.user_feedback_memory import UserFeedbackMemory
from app.agents.memory.agent_performance_memory import AgentPerformanceMemory
from app.agents.memory.agent_preference_memory import AgentPreferenceMemory
from app.agents.memory.tool_success_memory import ToolSuccessMemory
from app.db.session import get_db
from app.models.task import Task, Step, TaskStatus
from app.orchestrator.loop_v3 import execute_task_v3 as execute_task
from app.core.config import settings
from app.utils.file_manager import FileManager

logger = structlog.get_logger()
router = APIRouter()

# ── Singletons ────────────────────────────────────────────────────────────────
agent_performance_memory = AgentPerformanceMemory()
agent_preference_memory  = AgentPreferenceMemory()
tool_success_memory      = ToolSuccessMemory()
feedback_memory          = UserFeedbackMemory()


# ── Pydantic schemas ──────────────────────────────────────────────────────────

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
    tool_name: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True


class TaskResponse(BaseModel):
    task_id: str
    user_input: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    session_id: Optional[str] = None
    steps: List[StepResponse] = []

    class Config:
        from_attributes = True


# ── Helper ────────────────────────────────────────────────────────────────────

def _serialize_step(step: Step) -> StepResponse:
    return StepResponse(
        id=step.id,
        step_number=step.step_number,
        instruction=step.instruction,
        status=step.status,
        tool_name=step.tool_name,
        result=step.result,
        error=step.error,
        retry_count=step.retry_count,
        completed_at=step.completed_at.isoformat() if step.completed_at else None,
    )


def _serialize_task(task: Task, include_steps: bool = True) -> TaskResponse:
    return TaskResponse(
        task_id=task.id,
        user_input=task.user_input,
        status=task.status,
        created_at=task.created_at.isoformat(),
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
        error_message=task.error_message,
        session_id=task.session_id,
        steps=(
            [_serialize_step(s) for s in sorted(task.steps, key=lambda s: s.step_number)]
            if include_steps else []
        ),
    )


# ── Task routes ───────────────────────────────────────────────────────────────

@router.post("/tasks", response_model=TaskResponse)
async def create_task(
    task_data: TaskCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Create a new task and start execution in the background."""
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
    background_tasks.add_task(execute_task, task_id)

    return _serialize_task(task, include_steps=False)


@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None, description="Filter by status: PENDING|RUNNING|COMPLETED|FAILED"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    db: Session = Depends(get_db),
):
    """List tasks with optional status and session filters."""
    q = db.query(Task)
    if status:
        q = q.filter(Task.status == status.upper())
    if session_id:
        q = q.filter(Task.session_id == session_id)
    tasks = q.order_by(Task.created_at.desc()).offset(offset).limit(limit).all()
    return [_serialize_task(t) for t in tasks]


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, db: Session = Depends(get_db)):
    """Get full task detail including all steps."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return _serialize_task(task)


@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str, db: Session = Depends(get_db)):
    """
    Return just the final output of a completed task.
    Pulls from the last completed step's result — clean, no noise.
    """
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status == TaskStatus.PENDING or task.status == TaskStatus.RUNNING:
        return {"task_id": task_id, "status": task.status, "result": None}

    if task.status == TaskStatus.FAILED:
        return {
            "task_id": task_id,
            "status": task.status,
            "result": None,
            "error": task.error_message,
        }

    # Find the best result: last completed step with non-empty output
    completed_steps = sorted(
        [s for s in task.steps if s.status == "COMPLETED" and s.result],
        key=lambda s: s.step_number,
        reverse=True,
    )
    result = completed_steps[0].result if completed_steps else None

    return {
        "task_id": task_id,
        "status": task.status,
        "result": result,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
    }


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, db: Session = Depends(get_db)):
    """Cancel a running or pending task."""
    from app.utils.websocket_manager import cancellation_store, ws_manager

    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status not in [TaskStatus.PENDING, TaskStatus.RUNNING]:
        raise HTTPException(status_code=400, detail=f"Task is {task.status} — cannot cancel")

    cancellation_store.cancel(task_id)
    task.status = TaskStatus.FAILED
    task.error_message = "Cancelled by user"
    task.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    await ws_manager.emit(task_id, {"phase": "cancelled", "status": "cancelled"})
    logger.info("task_cancelled", task_id=task_id)
    return {"task_id": task_id, "status": "cancelled"}


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, db: Session = Depends(get_db)):
    """Delete a task and all its steps from the database."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Cannot delete a running task — cancel it first")

    db.delete(task)
    db.commit()
    logger.info("task_deleted", task_id=task_id)
    return {"task_id": task_id, "deleted": True}


# ── Session routes ────────────────────────────────────────────────────────────

@router.get("/sessions")
async def list_sessions(db: Session = Depends(get_db)):
    """List all unique session IDs with task counts and latest activity."""
    rows = (
        db.query(Task.session_id)
        .filter(Task.session_id.isnot(None))
        .distinct()
        .all()
    )
    session_ids = [r[0] for r in rows]

    sessions = []
    for sid in session_ids:
        tasks = (
            db.query(Task)
            .filter(Task.session_id == sid)
            .order_by(Task.created_at.desc())
            .all()
        )
        sessions.append({
            "session_id": sid,
            "task_count": len(tasks),
            "latest_task": tasks[0].user_input[:80] if tasks else "",
            "last_active": tasks[0].created_at.isoformat() if tasks else "",
            "statuses": {
                "completed": sum(1 for t in tasks if t.status == TaskStatus.COMPLETED),
                "failed":    sum(1 for t in tasks if t.status == TaskStatus.FAILED),
                "running":   sum(1 for t in tasks if t.status == TaskStatus.RUNNING),
            },
        })

    sessions.sort(key=lambda x: x["last_active"], reverse=True)
    return {"sessions": sessions, "total": len(sessions)}


# ── File routes ───────────────────────────────────────────────────────────────

def _get_fm() -> FileManager:
    return FileManager(base_dir=settings.WORKSPACE_DIR)


@router.get("/files")
async def list_files():
    """List all files in the shared workspace."""
    files = _get_fm().list_files()
    return {"files": sorted(files), "count": len(files)}


@router.get("/files/{filename}")
async def get_file(filename: str):
    """Read a file from the shared workspace."""
    content = _get_fm().read_file(filename)
    if content is None:
        raise HTTPException(status_code=404, detail="File not found")
    return PlainTextResponse(content=content)


@router.post("/files/{filename}", status_code=201)
async def write_file(filename: str, body: dict):
    """
    Write content to a file in the shared workspace.
    Body: {"content": "..."}
    """
    content = body.get("content", "")
    if not content:
        raise HTTPException(status_code=400, detail="'content' field is required")
    ok = _get_fm().write_file(filename, content)
    if not ok:
        raise HTTPException(status_code=500, detail=f"Failed to write {filename}")
    return {"filename": filename, "size": len(content), "written": True}


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a file from the shared workspace."""
    ok = _get_fm().delete_file(filename)
    if not ok:
        raise HTTPException(status_code=404, detail="File not found or could not be deleted")
    return {"filename": filename, "deleted": True}


# ── Traces ────────────────────────────────────────────────────────────────────

@router.get("/traces/{task_id}")
async def get_trace(task_id: str):
    """Return the execution trace JSON written by loop_v3 for a task."""
    trace_path = Path(settings.COSTS_DIR).parent / "traces" / f"task_{task_id}.json"
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail="Trace not found for this task")
    try:
        with open(trace_path) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read trace: {e}")


# ── Analytics ─────────────────────────────────────────────────────────────────

@router.get("/analytics")
async def get_analytics():
    """Aggregate metrics from all task cost-tracker files."""
    costs_dir = Path(settings.COSTS_DIR)
    files = list(costs_dir.glob("task_*.json"))

    if not files:
        return {"total_tasks": 0, "tasks": []}

    # Read cost files off the event loop thread — avoids blocking async workers
    def _load_files():
        results = []
        for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:50]:
            try:
                with open(f) as fp:
                    results.append(json.load(fp))
            except Exception as e:
                logger.warning("analytics_file_unreadable", file=str(f), error=str(e))
        return results

    tasks = await asyncio.to_thread(_load_files)

    total      = len(tasks)
    successful = sum(1 for t in tasks if t.get("success"))
    failed     = total - successful

    return {
        "total_tasks":    total,
        "successful":     successful,
        "failed":         failed,
        "success_rate":   round(successful / total * 100, 1) if total > 0 else 0,
        "avg_duration":   round(sum(t.get("duration_sec", 0) for t in tasks) / total, 1) if total else 0,
        "avg_llm_calls":  round(sum(t.get("total_llm_calls", 0) for t in tasks) / total, 1) if total else 0,
        "avg_cost_usd":   round(sum(t.get("estimated_cost_usd", 0) for t in tasks) / total, 6) if total else 0,
        "total_cost_usd": round(sum(t.get("estimated_cost_usd", 0) for t in tasks), 4),
        "avg_retries":    round(sum(t.get("total_retries", 0) for t in tasks) / total, 1) if total else 0,
        "tasks":          tasks[:20],
    }


# ── System ────────────────────────────────────────────────────────────────────

@router.get("/system/status")
async def system_status():
    """Return current system configuration and feature flags."""
    return {
        "version":           "1.0.0",
        "env":               settings.ENV,
        "features": {
            "python_executor": settings.ENABLE_PYTHON_EXECUTOR,
            "shell_executor":  settings.ENABLE_SHELL,
        },
        "workspace_dir":     settings.WORKSPACE_DIR,
        "costs_dir":         settings.COSTS_DIR,
    }


@router.get("/health")
def health():
    return {"status": "ok", "service": "agent-system"}


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Record user feedback (good/bad/neutral) for a query."""
    feedback_memory.record_feedback(query=request.query, feedback=request.feedback)
    return {"success": True}


@router.get("/feedback/stats")
async def feedback_stats():
    return feedback_memory.get_feedback_stats()


@router.get("/feedback/history")
async def feedback_history():
    return feedback_memory.feedback[-50:]


# ── Memory / agent stats ──────────────────────────────────────────────────────

@router.get("/memory/stats")
async def memory_stats():
    return {
        "feedback_entries":          len(feedback_memory.feedback),
        "agent_performance_entries": len(agent_performance_memory.all()),
        "agent_preference_entries":  len(agent_preference_memory.preferences),
        "tool_success_entries":      len(tool_success_memory.memory),
    }


@router.get("/agents/performance")
async def agent_performance():
    return agent_performance_memory.all()


@router.get("/agents/preferences")
async def agent_preferences():
    return agent_preference_memory.preferences


@router.get("/tools/performance")
async def tool_performance():
    return tool_success_memory.memory
