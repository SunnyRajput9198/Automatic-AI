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
    - what_failed, root_causes, improvement_suggestions, pattern_quality,
      suggested_action
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
        1. Explicit suggested_action from Reflection → use it directly
        2. Pattern quality very low → abort (unreliable pattern)
        3. Prompt/token/context issues → retry with smaller prompt
        4. Tool failure / wrong tool → switch agent
        5. Syntax / code error → retry
        6. Step is optional/irrelevant → skip
        7. Default → retry
        """

        # --- Path 1: explicit suggestion from the Reflection model
        action = (reflection_output.get("suggested_action") or "").strip()
        root_causes = reflection_output.get("root_causes", [])
        base_reason = root_causes[0] if root_causes else "Unknown failure"

        if action in self.VALID_ACTIONS:
            logger.info("recovery_explicit_action", action=action, reason=base_reason)
            return RecoveryDecision(action=action, reason=base_reason)

        # --- Path 2: derive from actual Reflection fields
        what_failed = reflection_output.get("what_failed", [])
        suggestions = reflection_output.get("improvement_suggestions", [])
        pattern_quality = float(reflection_output.get("pattern_quality", 0.5))

        # Combine all text for keyword scanning
        all_text = " ".join(what_failed + root_causes + suggestions).lower()

        # Very low pattern quality → whole approach is unreliable, abort
        if pattern_quality < 0.2:
            logger.info("recovery_abort_low_quality", quality=pattern_quality)
            return RecoveryDecision(
                action="abort_task",
                reason=f"Pattern quality too low ({pattern_quality}) to recover",
            )

        # Prompt/token/context issues → retry with a smaller prompt
        if any(k in all_text for k in [
            "prompt", "token", "too long", "context", "truncat"
        ]):
            logger.info("recovery_smaller_prompt", reason=base_reason)
            return RecoveryDecision(
                action="retry_with_smaller_prompt",
                reason=base_reason,
            )

        # Tool failure / wrong tool selection → switch to a different agent
        if any(k in all_text for k in [
            "tool", "executor", "tool failed", "tool not found",
            "wrong tool", "tool selection",
        ]):
            logger.info("recovery_switch_agent", reason=base_reason)
            return RecoveryDecision(
                action="switch_agent",
                reason=base_reason,
            )

        # Syntax / code errors → retry, executor may generate better code
        if any(k in all_text for k in [
            "syntax", "import", "module", "indentation", "nameerror",
        ]):
            logger.info("recovery_retry_code_error", reason=base_reason)
            return RecoveryDecision(
                action="retry",
                reason=base_reason,
            )

        # Step is optional or irrelevant → skip rather than keep retrying
        if any(k in all_text for k in [
            "skip", "not necessary", "optional", "irrelevant",
        ]):
            logger.info("recovery_skip_step", reason=base_reason)
            return RecoveryDecision(
                action="skip_step",
                reason=base_reason,
            )

        # Default: retry once more
        logger.info("recovery_default_retry", reason=base_reason)
        return RecoveryDecision(action="retry", reason=base_reason)
