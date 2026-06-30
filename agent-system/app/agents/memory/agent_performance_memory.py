from app.agents.memory.trust_store import calculate_trust
import json
import os
import structlog
from typing import Dict

from app.core.config import settings

logger = structlog.get_logger()

WORKSPACE = settings.SHARED_WORKSPACE
MEMORY_PATH = os.path.join(WORKSPACE, "agent_performance.json")

class AgentPerformanceMemory:
    """
    Stores and recalls agent performance across runs.
    Stats include a 'role' field so task_router can resolve
    agent names (e.g. 'researcher_001') back to roles ('researcher').
    A trust score (0-1) is computed and stored alongside raw stats
    using calculate_trust() from trust_store.
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
            logger.error("agent_performance_memory_corrupted", error=str(e))
            self.memory = {}

    def save(self):
        os.makedirs(WORKSPACE, exist_ok=True)
        with open(MEMORY_PATH, "w") as f:
            json.dump(self.memory, f, indent=2)

    def update(self, agent_name: str, stats: Dict):
        """
        Store stats for an agent.
        Injects 'role' derived from the agent name if not already present
        and computes a trust score via calculate_trust().
        """
        if "role" not in stats:
            role = agent_name.split("_")[0] if "_" in agent_name else agent_name
            stats = {**stats, "role": role}

        # Compute and persist trust score so task_router / coordinator can use it
        stats = {**stats, "trust_score": calculate_trust(stats)}

        self.memory[agent_name] = stats
        self.save()

        logger.info(
            "agent_performance_saved",
            agent=agent_name,
            role=stats.get("role"),
            success_rate=stats.get("success_rate"),
            trust_score=stats.get("trust_score"),
        )

    def get(self, agent_name: str):
        return self.memory.get(agent_name)

    def all(self):
        return self.memory
