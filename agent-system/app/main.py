from fastapi import FastAPI
from contextlib import asynccontextmanager
import structlog
from fastapi.middleware.cors import CORSMiddleware
from app.db.session import init_db
from app.api import tasks
from app.api import health
from app.core.config import settings
from fastapi import WebSocket, WebSocketDisconnect
from app.utils.websocket_manager import ws_manager
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    logger.info("Initializing database...")
    init_db()
    logger.info(
        "application_started",
        env=settings.ENV,
        shell_enabled=settings.ENABLE_SHELL,
        python_enabled=settings.ENABLE_PYTHON_EXECUTOR
    )
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Autonomous Agent System",
    description="Production-grade multi-agent system",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(health.router)
app.include_router(tasks.router, prefix="/api/v1", tags=["tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React/Next frontend
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "Autonomous Agent System API",
        "version": "1.0.0",
        "endpoints": {
        "health": "GET /health",
            "create_task": "POST /api/v1/tasks",
            "get_task": "GET /api/v1/tasks/{task_id}",
            "list_tasks": "GET /api/v1/tasks"
        }
    }

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await ws_manager.connect(task_id, websocket)
    try:
        while True:
            # Keep connection alive, just receive any pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(task_id)