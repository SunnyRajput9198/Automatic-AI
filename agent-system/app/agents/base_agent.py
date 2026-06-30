from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()


class AgentResult(BaseModel):
    """Standard result format from any agent"""

    success: bool
    output: str
    metadata: Dict[str, Any] = {}
    confidence: float = 0.5
    errors: List[str] = []
    agent_name: str
    duration_sec: float = 0.0


class BaseAgent(ABC):
    """
    Base class for all specialist agents.

    Stats (call_count, success_count, failure_count) are persisted to
    AgentPerformanceMemory on every record_success / record_failure call
    so they survive process restarts regardless of whether the agent is
    called via the coordinator or directly (e.g. from the research graph).
    """

    def __init__(self, name: str, role: str, allowed_tools: List[str]):
        self.name          = name
        self.role          = role
        self.allowed_tools = allowed_tools
        self.call_count    = 0
        self.success_count = 0
        self.failure_count = 0

        # Restore persisted counts so in-memory stats are warm on restart
        self._restore_stats()

        logger.info(
            "agent_initialized",
            name=self.name,
            role=self.role,
            allowed_tools=self.allowed_tools,
        )

    def _restore_stats(self) -> None:
        """Load previously persisted stats from AgentPerformanceMemory."""
        try:
            from app.agents.memory.agent_performance_memory import AgentPerformanceMemory
            mem = AgentPerformanceMemory()
            saved = mem.get(self.name)
            if saved:
                self.call_count    = int(saved.get("calls", 0))
                self.success_count = int(saved.get("successes", 0))
                self.failure_count = int(saved.get("failures", 0))
        except Exception:
            pass  # Non-fatal — fresh counts are fine if memory is unavailable

    def _persist_stats(self) -> None:
        """Push current stats to AgentPerformanceMemory."""
        try:
            from app.agents.memory.agent_performance_memory import AgentPerformanceMemory
            AgentPerformanceMemory().update(self.name, self.get_stats())
        except Exception:
            pass  # Non-fatal — don't crash the agent on a memory write failure

    @abstractmethod
    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        pass

    def can_use_tool(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools

    def record_success(self):
        self.call_count    += 1
        self.success_count += 1
        self._persist_stats()

    def record_failure(self):
        self.call_count    += 1
        self.failure_count += 1
        self._persist_stats()

    def get_success_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name":         self.name,
            "role":         self.role,
            "calls":        self.call_count,
            "successes":    self.success_count,
            "failures":     self.failure_count,
            "success_rate": self.get_success_rate(),
        }
