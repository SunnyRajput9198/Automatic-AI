import structlog
from typing import Dict, Any

logger = structlog.get_logger()


class RecoveryDecision:
    def __init__(self, action: str, reason: str = ""):
        self.action = action
        self.reason = reason


class RecoveryManager:
    """
    Converts reflection output into a concrete recovery action.
    Reads the fields that Reflection model actually produces:
    - what_failed, root_causes, improvement_suggestions, pattern_quality
    """

    VALID_ACTIONS = {
        "retry",
        "retry_with_smaller_prompt",
        "switch_agent",
        "skip_step",
        "abort_task",
    }

    def decide(self, reflection_output: Dict[str, Any]) -> RecoveryDecision:
        """
        Derive a recovery action from reflection output.

        Priority order:
        1. If reflection explicitly suggests an action → use it
        2. If pattern_quality is very low → abort (unreliable pattern)
        3. If root cause mentions prompt/token issues → retry with smaller prompt
        4. If root cause mentions tool failure → switch agent
        5. Default → retry
        """

        # --- Path 1: explicit suggestion (future-proof if we add the field later)
        action = reflection_output.get("suggested_action", "").strip()
        reason = reflection_output.get("failure_reason", "").strip()

        if action in self.VALID_ACTIONS:
            logger.info("recovery_explicit_action", action=action, reason=reason)
            return RecoveryDecision(action=action, reason=reason)

        # --- Path 2: derive from actual Reflection fields
        what_failed = reflection_output.get("what_failed", [])
        root_causes = reflection_output.get("root_causes", [])
        suggestions = reflection_output.get("improvement_suggestions", [])
        pattern_quality = float(reflection_output.get("pattern_quality", 0.5))

        # Combine all text for keyword scanning
        all_text = " ".join(
            what_failed + root_causes + suggestions
        ).lower()

        reason = root_causes[0] if root_causes else "Unknown failure"

        # Very low pattern quality = this whole approach is unreliable
        if pattern_quality < 0.2:
            logger.info("recovery_abort_low_quality", quality=pattern_quality)
            return RecoveryDecision(
                action="abort_task",
                reason=f"Pattern quality too low ({pattern_quality}) to recover"
            )

        # Prompt/token/context issues → retry with smaller prompt
        if any(k in all_text for k in [
            "prompt", "token", "too long", "context", "truncat"
        ]):
            logger.info("recovery_smaller_prompt", reason=reason)
            return RecoveryDecision(
                action="retry_with_smaller_prompt",
                reason=reason
            )

        # Tool failure / tool not found → switch to a different agent
        if any(k in all_text for k in [
            "tool", "executor", "tool failed", "tool not found",
            "wrong tool", "tool selection"
        ]):
            logger.info("recovery_switch_agent", reason=reason)
            return RecoveryDecision(
                action="switch_agent",
                reason=reason
            )

        # Syntax / code error → retry, executor might generate better code
        if any(k in all_text for k in [
            "syntax", "import", "module", "indentation", "nameerror"
        ]):
            logger.info("recovery_retry_code_error", reason=reason)
            return RecoveryDecision(
                action="retry",
                reason=reason
            )

        # Default: retry once more
        logger.info("recovery_default_retry", reason=reason)
        return RecoveryDecision(action="retry", reason=reason)