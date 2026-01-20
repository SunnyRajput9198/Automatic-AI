import structlog
from typing import Dict, Any

logger = structlog.get_logger()


class RecoveryDecision:
    """
    Normalized recovery decision.
    """

    def __init__(self, action: str, reason: str = ""):
        self.action = action
        self.reason = reason


class RecoveryManager:
    """
    Converts reflection output into real system actions.

    This is the core of self-healing behavior.
    """

    VALID_ACTIONS = {
        "retry",
        "retry_with_smaller_prompt",
        "switch_agent",
        "skip_step",
        "abort_task",
        "switch_agent"
    }

    def decide(self, reflection_output: Dict[str, Any]) -> RecoveryDecision:
        """
        Decide what the system should do next.

        Args:
            reflection_output: parsed JSON from ReflectionAgent

        Returns:
            RecoveryDecision
        """

        action = reflection_output.get("suggested_action", "").strip()
        reason = reflection_output.get("failure_reason", "")

        if action not in self.VALID_ACTIONS:
            logger.warning(
                "invalid_recovery_action",
                action=action,
                fallback="abort_task",
            )
            return RecoveryDecision(
                action="abort_task",
                reason="Invalid recovery action",
            )

        logger.info(
            "recovery_decision_made",
            action=action,
            reason=reason,
        )

        return RecoveryDecision(
            action=action,
            reason=reason,
        )
