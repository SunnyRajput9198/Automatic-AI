import json
import os
import structlog
from typing import Dict

from app.core.config import settings

logger = structlog.get_logger()

WORKSPACE = settings.SHARED_WORKSPACE
MEMORY_PATH = os.path.join(
    WORKSPACE,
    "tool_success_memory.json"
)


class ToolSuccessMemory:
    """
    Stores success/failure counts per tool.
    """

    def __init__(self):
        self.memory: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(MEMORY_PATH):
                with open(MEMORY_PATH, "r") as f:
                    self.memory = json.load(f)
            else:
                self.memory = {}
        except Exception as e:
            logger.error("tool_success_memory_corrupted", error=str(e))
            self.memory = {}

    def save(self):
        os.makedirs(WORKSPACE, exist_ok=True)

        with open(MEMORY_PATH, "w") as f:
            json.dump(self.memory, f, indent=2)

    def record_success(self, tool_name: str):
        stats = self.memory.setdefault(
            tool_name,
            {
                "success": 0,
                "failure": 0,
            },
        )

        stats["success"] += 1

        self.save()

        logger.info(
            "tool_success_recorded",
            tool=tool_name,
            success=stats["success"],
        )

    def record_failure(self, tool_name: str):
        stats = self.memory.setdefault(
            tool_name,
            {
                "success": 0,
                "failure": 0,
            },
        )

        stats["failure"] += 1

        self.save()

        logger.info(
            "tool_failure_recorded",
            tool=tool_name,
            failure=stats["failure"],
        )

    def get_stats(self, tool_name: str):
        return self.memory.get(tool_name)

    def top_tools(self):
        ranked = []

        for tool, stats in self.memory.items():
            total = (
                stats["success"] +
                stats["failure"]
            )

            score = (
                stats["success"] / total
                if total > 0
                else 0
            )

            ranked.append(
                (tool, score)
            )

        ranked.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        return [x[0] for x in ranked[:5]]