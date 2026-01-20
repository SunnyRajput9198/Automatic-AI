from fastapi import FastAPI
from contextlib import asynccontextmanager
import structlog

from app.db.session import init_db
from app.api import tasks

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Autonomous Agent System",
    description="Production-grade multi-agent system",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(tasks.router, prefix="/api/v1", tags=["tasks"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agent-system"}

@app.get("/")
async def root():
    return {
        "message": "Autonomous Agent System API",
        "version": "1.0.0",
        "endpoints": {
            "create_task": "POST /api/v1/tasks",
            "get_task": "GET /api/v1/tasks/{task_id}",
            "list_tasks": "GET /api/v1/tasks"
        }
    }
    