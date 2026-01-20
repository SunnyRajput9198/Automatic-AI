import structlog
from typing import Dict, Any
from app.agents.base_agent import BaseAgent

logger = structlog.get_logger()


class AgentSwitcher:
    """
    Handles dynamic agent switching when a step fails.
    """

    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents

    async def switch_and_execute(
        self,
        failed_agent: str,
        instruction: str,
        context: Dict[str, Any],
    ):
        """
        Pick a different agent and retry execution.
        """

        for role, agent in self.agents.items():
            if role == failed_agent:
                continue

            logger.info(
                "agent_switch_attempt",
                from_agent=failed_agent,
                to_agent=role
            )

            try:
                result = await agent.execute(instruction, context)

                if result.success:
                    logger.info(
                        "agent_switch_success",
                        agent=role
                    )
                    return result, role

            except Exception as e:
                logger.error(
                    "agent_switch_failed",
                    agent=role,
                    error=str(e)
                )

        return None, None
