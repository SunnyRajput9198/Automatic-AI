import structlog
from typing import Dict, Any, Optional, Tuple
from app.agents.base_agent import BaseAgent, AgentResult

logger = structlog.get_logger()


class AgentSwitcher:
    """
    Tries each available specialist agent until one succeeds.
    Used when the primary executor fails and recovery decides to switch.
    """

    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents

    async def switch_and_execute(
        self,
        failed_agent: str,
        instruction: str,
        context: Dict[str, Any],
    ) -> Tuple[Optional[AgentResult], Optional[str]]:
        """
        Try each agent except the one that already failed.

        Args:
            failed_agent: role name of agent that failed (skip this one)
            instruction:  the step instruction to retry
            context:      current execution context

        Returns:
            (AgentResult, role_name) if one succeeds
            (None, None) if all fail
        """

        if not self.agents:
            logger.warning("agent_switcher_no_agents")
            return None, None

        for role, agent in self.agents.items():

            # FIX: also skip if the agent's name matches, not just role
            # because loop_v3 passes "executor" as failed_agent but
            # WEEK4_AGENTS uses "researcher"/"engineer"/"writer" as roles
            # This means we try all specialist agents when executor fails
            # which is the correct intended behaviour
            if role == failed_agent or agent.name == failed_agent:
                logger.debug("agent_switcher_skipping", role=role)
                continue

            logger.info(
                "agent_switch_attempt",
                from_agent=failed_agent,
                to_agent=role
            )

            try:
                result = await agent.execute(instruction, context)

                if result.success:
                    logger.info("agent_switch_success", agent=role)
                    return result, role
                else:
                    logger.warning(
                        "agent_switch_unsuccessful",
                        agent=role,
                        output_preview=result.output[:100] if result.output else ""
                    )

            except Exception as e:
                logger.error(
                    "agent_switch_failed",
                    agent=role,
                    error=str(e)
                )
                continue  # FIX: was implicitly continuing but now explicit

        logger.error(
            "agent_switcher_all_failed",
            failed_agent=failed_agent,
            tried=list(self.agents.keys())
        )
        return None, None