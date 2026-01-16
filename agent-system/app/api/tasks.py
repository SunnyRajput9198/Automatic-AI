from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import uuid
import structlog

from app.db.session import get_db
from app.models.task import Task, Step, TaskStatus
from app.orchestrator.loop import execute_task

logger = structlog.get_logger()
router = APIRouter()

class TaskCreate(BaseModel):
    task: str

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
    steps: List[StepResponse] = []

    class Config:
        from_attributes = True

@router.post("/tasks", response_model=TaskResponse)
async def create_task(
    task_data: TaskCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new task and start execution"""
    task_id = str(uuid.uuid4())
    
    task = Task(
        id=task_id,
        user_input=task_data.task,
        status=TaskStatus.PENDING
    )
    
    db.add(task)
    db.commit()
    db.refresh(task)
    
    logger.info("task_created", task_id=task_id, user_input=task_data.task)
    
    # Execute task in background
    background_tasks.add_task(execute_task, task_id)
    
    return TaskResponse(
        task_id=task.id,
        user_input=task.user_input,
        status=task.status,
        created_at=task.created_at.isoformat(),
        steps=[]
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
            retry_count=step.retry_count
        )
        for step in sorted(task.steps, key=lambda s: s.step_number)
    ]
    
    return TaskResponse(
        task_id=task.id,
        user_input=task.user_input,
        status=task.status,
        created_at=task.created_at.isoformat(),
        steps=steps
    )

@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List all tasks"""
    tasks = db.query(Task).order_by(Task.created_at.desc()).offset(offset).limit(limit).all()
    
    return [
        TaskResponse(
            task_id=task.id,
            user_input=task.user_input,
            status=task.status,
            created_at=task.created_at.isoformat(),
            steps=[
                StepResponse(
                    id=step.id,
                    step_number=step.step_number,
                    instruction=step.instruction,
                    status=step.status,
                    result=step.result,
                    error=step.error,
                    retry_count=step.retry_count
                )
                for step in sorted(task.steps, key=lambda s: s.step_number)
            ]
        )
        for task in tasks
    ]