import json
import os
import structlog
from typing import Dict

logger = structlog.get_logger()

FAIL_PATH = "workspace/tool_failures.json"


class ToolFailureMemory:
    def __init__(self):
        self.failures: Dict[str, int] = {}
        self._load()

    def _load(self):
        try:
            if os.path.exists(FAIL_PATH):
                with open(FAIL_PATH, "r") as f:
                    self.failures = json.load(f)
        except Exception as e:
            logger.error("tool_failure_memory_corrupted", error=str(e))
            self.failures = {}

    def _save(self):
        os.makedirs("workspace", exist_ok=True)
        with open(FAIL_PATH, "w") as f:
            json.dump(self.failures, f, indent=2)

    def record_failure(self, tool_name: str):
        self.failures[tool_name] = self.failures.get(tool_name, 0) + 1
        self._save()

        logger.info(
            "tool_failure_recorded",
            tool=tool_name,
            count=self.failures[tool_name],
        )

    def should_avoid(self, tool_name: str, threshold: int = 2) -> bool:
        return self.failures.get(tool_name, 0) >= threshold
