import json
import os
import re
import structlog
from typing import Dict, Optional

logger = structlog.get_logger()

from app.core.config import settings
WORKSPACE = settings.SHARED_WORKSPACE
PREF_PATH = os.path.join(WORKSPACE, "agent_preferences.json")

# Common stopwords that add no signal to task keys
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "me", "my", "we", "our", "you",
    "your", "it", "its", "that", "this", "these", "those", "what", "how",
    "why", "when", "where", "who", "which",
}


class AgentPreferenceMemory:
    def __init__(self):
        self.preferences: Dict[str, str] = {}
        self._load()

    def _load(self):
        if os.path.exists(PREF_PATH):
            try:
                with open(PREF_PATH, "r") as f:
                    self.preferences = json.load(f)
            except Exception as e:
                logger.error("agent_preference_memory_corrupted", error=str(e))
                self.preferences = {}
        else:
            self.preferences = {}

    def _save(self):
        os.makedirs(WORKSPACE, exist_ok=True)
        with open(PREF_PATH, "w") as f:
            json.dump(self.preferences, f, indent=2)

    def record_success(self, task_description: str, agent_name: str):
        key = self._task_key(task_description)
        self.preferences[key] = agent_name
        self._save()
        logger.info("agent_preference_learned", task_key=key, agent=agent_name)

    def get_preferred_agent(self, task_description: str) -> Optional[str]:
        key = self._task_key(task_description)
        agent = self.preferences.get(key)

        valid_agents = {"researcher", "engineer", "writer"}

        if agent is None:
            return None

        if agent not in valid_agents:
            logger.warning("invalid_agent_preference", key=key, agent=agent)
            return None

        return agent

    def _task_key(self, task: str) -> str:
        """
        Build a stable key from the task description.
        - Lowercased and stripped of punctuation
        - Stopwords removed so 'research the FastAPI' and
          'research FastAPI' map to the same key
        - First 6 meaningful words used (more specific than 4)
        """
        words = re.sub(r"[^a-z0-9\s]", "", task.lower()).split()
        meaningful = [w for w in words if w not in _STOPWORDS]
        return " ".join(meaningful[:6])
