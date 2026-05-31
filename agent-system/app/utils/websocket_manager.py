import asyncio
import json
import structlog
from typing import Dict, Any
from fastapi import WebSocket

logger = structlog.get_logger()


class WebSocketManager:
    """
    Manages WebSocket connections per task.
    Each task_id can have one active WebSocket connection.
    """

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}

    async def connect(self, task_id: str, websocket: WebSocket):
        await websocket.accept()
        self.connections[task_id] = websocket
        logger.info("websocket_connected", task_id=task_id)

    def disconnect(self, task_id: str):
        if task_id in self.connections:
            del self.connections[task_id]
            logger.info("websocket_disconnected", task_id=task_id)

    async def emit(self, task_id: str, event: Dict[str, Any]):
        """Send event to client if connected."""
        websocket = self.connections.get(task_id)
        if not websocket:
            return
        try:
            await websocket.send_text(json.dumps(event))
        except Exception as e:
            logger.warning("websocket_send_failed", task_id=task_id, error=str(e))
            self.disconnect(task_id)


# Global instance
ws_manager = WebSocketManager()