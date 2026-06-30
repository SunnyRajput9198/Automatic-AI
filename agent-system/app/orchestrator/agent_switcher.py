import structlog
from typing import Dict, Any, Optional, Tuple, Set

from app.agents.base_agent import BaseAgent, AgentResult

logger = structlog.get_logger()

# Hard cap: never try more than this many agents in a single switch call.
# Prevents A→B→A→B infinite cycling when all agents keep failing.
MAX_SWITCHES = 3


class AgentSwitcher:
    """
    Tries each available specialist agent until one succeeds or the switch
    cap is reached. Used when the primary executor fails and RecoveryManager
    decides to switch agents.

    Guarantees:
    - Never tries the same agent twice in one call (tried_roles set).
    - Never exceeds MAX_SWITCHES attempts regardless of pool size.
    - Logs every attempt and the final outcome for observability.
    """

    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents

    async def switch_and_execute(
        self,
        failed_agent: str,
        instruction: str,
        context: Dict[str, Any],
        already_tried: Optional[Set[str]] = None,
    ) -> Tuple[Optional[AgentResult], Optional[str]]:
        """
        Try each agent except those already known to have failed.

        Args:
            failed_agent:   role / name of agent that just failed (always skipped)
            instruction:    the step instruction to retry
            context:        current execution context
            already_tried:  optional set of role names tried in prior switch
                            calls for this step — prevents cross-call cycling

        Returns:
            (AgentResult, role_name) if an agent succeeds
            (None, None)             if all candidates fail or cap is reached
        """
        if not self.agents:
            logger.warning("agent_switcher_no_agents")
            return None, None

        # Build the exclusion set: failed agent + anything tried before
        skip: Set[str] = {failed_agent}
        if already_tried:
            skip.update(already_tried)

        candidates = [
            (role, agent)
            for role, agent in self.agents.items()
            if role not in skip and agent.name not in skip
        ]

        if not candidates:
            logger.warning(
                "agent_switcher_no_candidates",
                failed_agent=failed_agent,
                skipped=list(skip),
            )
            return None, None

        # Enforce the switch cap
        candidates = candidates[:MAX_SWITCHES]
        logger.info(
            "agent_switcher_starting",
            failed_agent=failed_agent,
            candidates=[r for r, _ in candidates],
            max_switches=MAX_SWITCHES,
        )

        for role, agent in candidates:
            logger.info("agent_switch_attempt", from_agent=failed_agent, to_agent=role)

            try:
                result = await agent.execute(instruction, context)

                if result.success:
                    logger.info("agent_switch_success", agent=role)
                    return result, role

                logger.warning(
                    "agent_switch_unsuccessful",
                    agent=role,
                    output_preview=(result.output or "")[:100],
                    errors=result.errors,
                )

            except Exception as e:
                logger.error("agent_switch_exception", agent=role, error=str(e))
                continue

        logger.error(
            "agent_switcher_all_failed",
            failed_agent=failed_agent,
            tried=[r for r, _ in candidates],
        )
        return None, None
