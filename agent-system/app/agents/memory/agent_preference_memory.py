import json
import os
import structlog
from typing import Dict

logger = structlog.get_logger()

PREF_PATH = "workspace/agent_preferences.json"


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
                logger.error(
                    "agent_preference_memory_corrupted",
                    error=str(e)
                )
                self.preferences = {}
        else:
            self.preferences = {}


    def _save(self):
        os.makedirs("workspace", exist_ok=True)
        with open(PREF_PATH, "w") as f:
            json.dump(self.preferences, f, indent=2)

    def record_success(self, task_description: str, agent_name: str):
        key = self._task_key(task_description)
        self.preferences[key] = agent_name
        self._save()

        logger.info(
            "agent_preference_learned",
            task_key=key,
            agent=agent_name
        )

    def get_preferred_agent(self, task_description: str):
        key = self._task_key(task_description)
        return self.preferences.get(key)

    def _task_key(self, task: str) -> str:
        return task.lower().split(" ")[0:4].__str__()
