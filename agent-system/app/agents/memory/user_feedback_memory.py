import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Optional

import structlog

from app.core.config import settings

logger = structlog.get_logger()

WORKSPACE = settings.SHARED_WORKSPACE
MEMORY_PATH = os.path.join(WORKSPACE, "user_feedback.json")


class UserFeedbackMemory:
    """
    Stores user feedback on agent responses.

    Example:
    {
        "query": "research react",
        "feedback": "good",
        "created_at": "2026-06-13T18:00:00"
    }
    """

    def __init__(self):
        self.feedback: List[Dict] = []
        self._load()

    def _load(self):
        try:
            if os.path.exists(MEMORY_PATH):
                with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                    self.feedback = json.load(f)
            else:
                self.feedback = []
        except Exception as e:
            logger.error("feedback_memory_load_failed", error=str(e))
            self.feedback = []

    def save(self):
        try:
            os.makedirs(WORKSPACE, exist_ok=True)

            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(self.feedback, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error("feedback_memory_save_failed", error=str(e))

    def record_feedback(
        self,
        query: str,
        feedback: str,
        answer: Optional[str] = None,
    ):
        """
        feedback: good | bad | neutral
        Appends to in-memory list and flushes to disk immediately
        (feedback volume is low, so per-call save is acceptable here).
        """
        item = {
            "query":      query,
            "feedback":   feedback.lower().strip(),
            "answer":     answer or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.feedback.append(item)
        self.save()
        logger.info("user_feedback_recorded", query=query[:100], feedback=feedback)

    def get_feedback_stats(self) -> Dict:
        total = len(self.feedback)

        good = sum(1 for f in self.feedback if f["feedback"] == "good")

        bad = sum(1 for f in self.feedback if f["feedback"] == "bad")

        neutral = sum(1 for f in self.feedback if f["feedback"] == "neutral")

        return {
            "total": total,
            "good": good,
            "bad": bad,
            "neutral": neutral,
            "positive_rate": (round(good / total, 2) if total > 0 else 0),
        }

    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        return self.feedback[-limit:]

    def get_feedback_for_query(self, query: str) -> List[Dict]:
        query_lower = query.lower()

        return [f for f in self.feedback if query_lower in f["query"].lower()]

    def clear(self):
        self.feedback = []
        self.save()

        logger.info("feedback_memory_cleared")
