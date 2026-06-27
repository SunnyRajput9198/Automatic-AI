import json
import os
import time
import structlog
from typing import Dict

logger = structlog.get_logger()

from app.core.config import settings
WORKSPACE = settings.SHARED_WORKSPACE
FAIL_PATH = os.path.join(WORKSPACE, "tool_failures.json")

# Failures older than this many hours are ignored (auto-expire)
FAILURE_TTL_HOURS = 6


class ToolFailureMemory:
    """
    Tracks consecutive tool failures with TTL-based expiry.
    Each entry stores the failure count and the timestamp of the
    most recent failure so stale entries don't permanently block tools.
    """

    def __init__(self):
        # { tool_name: {"count": int, "last_failure_ts": float} }
        self.failures: Dict[str, Dict] = {}
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
        os.makedirs(WORKSPACE, exist_ok=True)
        with open(FAIL_PATH, "w") as f:
            json.dump(self.failures, f, indent=2)

    def _is_expired(self, entry: Dict) -> bool:
        """Return True if the failure record is older than FAILURE_TTL_HOURS."""
        last_ts = entry.get("last_failure_ts", 0.0)
        age_hours = (time.time() - last_ts) / 3600
        return age_hours >= FAILURE_TTL_HOURS

    def record_failure(self, tool_name: str):
        entry = self.failures.get(tool_name, {"count": 0, "last_failure_ts": 0.0})
        # Reset count if previous failures have expired
        if self._is_expired(entry):
            entry["count"] = 0
        entry["count"] += 1
        entry["last_failure_ts"] = time.time()
        self.failures[tool_name] = entry
        self._save()
        logger.info(
            "tool_failure_recorded",
            tool=tool_name,
            count=entry["count"],
        )

    def should_avoid(self, tool_name: str, threshold: int = 2) -> bool:
        """
        Returns True only if the tool has >= threshold recent failures
        (within the TTL window).
        """
        entry = self.failures.get(tool_name)
        if not entry:
            return False
        if self._is_expired(entry):
            return False
        return entry.get("count", 0) >= threshold

    def reset_failures(self, tool_name: str):
        """Reset failure record after a successful execution."""
        if tool_name in self.failures:
            del self.failures[tool_name]
            self._save()
            logger.info("tool_failure_reset", tool=tool_name)
