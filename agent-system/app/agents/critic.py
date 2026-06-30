import structlog
from enum import Enum
from typing import Optional

from app.utils.json_parser import extract_json
from app.tools.base import ToolResult
from app.utils.llm import call_openai_with_tools

logger = structlog.get_logger()


class Verdict(str, Enum):
    PASS  = "PASS"
    RETRY = "RETRY"
    FAIL  = "FAIL"


class CriticResult:
    """Result of critic evaluation."""

    def __init__(
        self,
        verdict: Verdict,
        reason: str,
        suggestions: str = "",
        relevance_score: int = 0,
    ):
        self.verdict         = verdict
        self.reason          = reason
        self.suggestions     = suggestions
        self.relevance_score = relevance_score  # 0-100, logged for tuning


# ---------------------------------------------------------------------------
# Tool schema the LLM must call — enforces structured output, no free-text
# ---------------------------------------------------------------------------

_EVALUATE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_evaluation",
        "description": "Submit the structured evaluation of a step execution.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["PASS", "RETRY", "FAIL"],
                    "description": "PASS=success, RETRY=try differently, FAIL=give up",
                },
                "relevance_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "How well the output matches the instruction subject (0-100)",
                },
                "reason": {
                    "type": "string",
                    "description": "Detailed explanation of the verdict",
                },
                "suggestions": {
                    "type": "string",
                    "description": (
                        "REQUIRED when verdict=RETRY. "
                        "Must be a concrete improved search query or action, "
                        "e.g. 'try query: FastAPI vs Django async benchmarks 2024'. "
                        "Empty string for PASS/FAIL."
                    ),
                },
            },
            "required": ["verdict", "relevance_score", "reason", "suggestions"],
        },
    },
}

_SYSTEM_PROMPT = """You are a critical evaluator agent. Evaluate whether a step execution succeeded.

VERDICT OPTIONS:
- PASS:  Step completed. Output is on-topic and useful.
- RETRY: Step failed but a differently-worded query/approach could fix it.
- FAIL:  Step failed and retrying will not help.

RELEVANCE SCORE GUIDE (judge CONTENT, not fluency):
- 65-100: Output is about the exact subject → PASS
- 40-64 : Partially relevant, misses core subject → RETRY
- 0-39  : Wrong domain entirely → RETRY (first attempt) or FAIL (later)

EVALUATION PRIORITY ORDER:
1. TOPICAL RELEVANCE — Is the output actually about the subject in the instruction?
    A result sharing a keyword but covering a completely different domain is OFF-TOPIC.
2. Tool executed without errors.
3. Output is useful for subsequent steps.

KEY RULES:
- relevance_score < 65 is NOT a PASS, regardless of fluency or output length.
- Default to RETRY on first attempt (retry_count=0) for off-topic results before FAIL.
- When RETRY: suggestions MUST contain a concrete improved query, not vague advice.
- Only use FAIL on first attempt for fundamentally unsearchable topics.
- On retry_count >= 1 with same off-topic result: use FAIL.

Call the submit_evaluation function with your assessment."""


class CriticAgent:
    """
    Evaluates step execution and decides the next action.

    Uses tool binding to force structured output — no free-text JSON parsing.
    Threshold: relevance_score >= 65 for PASS (was 80, lowered to reduce
    false RETRYs while data is being collected for further tuning).
    """

    MAX_RETRIES    = 2
    PASS_THRESHOLD = 65   # tune upward once real score distribution is known

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model

    async def evaluate(
        self,
        step_instruction: str,
        tool_result: ToolResult,
        retry_count: int = 0,
    ) -> CriticResult:

        logger.info(
            "critic_evaluating",
            instruction=step_instruction[:80],
            success=tool_result.success,
            retry_count=retry_count,
        )

        # Hard stop: exceeded retries
        if retry_count >= self.MAX_RETRIES:
            return CriticResult(
                verdict=Verdict.FAIL,
                reason=f"Exceeded maximum retries ({self.MAX_RETRIES})",
                suggestions="",
                relevance_score=0,
            )

        try:
            metadata_str = str(tool_result.metadata)
        except Exception:
            metadata_str = "(unserializable metadata)"

        user_prompt = (
            f"STEP INSTRUCTION:\n{step_instruction}\n\n"
            f"TOOL EXECUTION:\n"
            f"- Success: {tool_result.success}\n"
            f"- Output: {tool_result.output[:500] if tool_result.output else '(empty)'}\n"
            f"- Error: {tool_result.error or '(none)'}\n"
            f"- Metadata: {metadata_str}\n\n"
            f"RETRY COUNT: {retry_count}/{self.MAX_RETRIES}\n\n"
            "Call submit_evaluation with your assessment."
        )

        try:
            result = await call_openai_with_tools(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tools=[_EVALUATE_TOOL],
                model=self.model,
                temperature=0.1,
                max_tokens=1000,
            )

            # ------------------------------------------------------------------
            # Tool binding path — structured, no parsing needed
            # ------------------------------------------------------------------
            if result["type"] == "tool_call" and result["name"] == "submit_evaluation":
                args            = result["arguments"]
                verdict_raw     = args.get("verdict", "FAIL").upper()
                relevance_score = int(args.get("relevance_score", 0))
                reason          = args.get("reason", "No reason provided")
                suggestions     = args.get("suggestions", "")

                try:
                    verdict = Verdict(verdict_raw)
                except ValueError:
                    verdict = Verdict.FAIL

                # Override: model said PASS but score is below threshold
                if verdict == Verdict.PASS and relevance_score < self.PASS_THRESHOLD:
                    logger.warning(
                        "critic_threshold_override",
                        score=relevance_score,
                        threshold=self.PASS_THRESHOLD,
                        original_verdict="PASS",
                    )
                    verdict = Verdict.RETRY if retry_count < self.MAX_RETRIES else Verdict.FAIL
                    reason = (
                        f"{reason} "
                        f"(overridden: score {relevance_score} < threshold {self.PASS_THRESHOLD})"
                    )

                # Safety: RETRY without a concrete suggestion → FAIL
                if verdict == Verdict.RETRY and not suggestions.strip():
                    logger.warning("critic_retry_missing_suggestions", reason=reason)
                    verdict = Verdict.FAIL
                    reason  = f"{reason} (RETRY had no concrete suggestion; treating as FAIL)"

                logger.info(
                    "critic_evaluated",
                    verdict=verdict,
                    relevance_score=relevance_score,
                    reason=reason[:120],
                )
                return CriticResult(
                    verdict=verdict,
                    reason=reason,
                    suggestions=suggestions,
                    relevance_score=relevance_score,
                )

            # ------------------------------------------------------------------
            # Fallback: model returned plain text instead of tool call
            # (shouldn't happen with tool binding, but handle gracefully)
            # ------------------------------------------------------------------
            logger.warning(
                "critic_tool_binding_fallback",
                preview=str(result.get("content", ""))[:200],
            )
            evaluation = extract_json(result.get("content", ""), context="critic")
            if evaluation:
                return self._from_dict(evaluation, retry_count)

            # Complete parse failure
            logger.error("critic_json_error", error="No valid response from tool binding or fallback")
            return CriticResult(
                verdict=Verdict.RETRY if retry_count < self.MAX_RETRIES else Verdict.FAIL,
                reason="Failed to parse critic evaluation",
                suggestions="",
                relevance_score=0,
            )

        except Exception as e:
            logger.error("critic_error", error=str(e))
            return CriticResult(
                verdict=Verdict.RETRY if retry_count < self.MAX_RETRIES else Verdict.FAIL,
                reason=f"Evaluation error: {str(e)}",
                suggestions="",
                relevance_score=0,
            )

    def _from_dict(self, data: dict, retry_count: int) -> CriticResult:
        """Build CriticResult from a raw dict (fallback path)."""
        verdict_raw     = data.get("verdict", "FAIL").upper()
        relevance_score = int(data.get("relevance_score", 0))
        reason          = data.get("reason", "No reason provided")
        suggestions     = data.get("suggestions", "")

        try:
            verdict = Verdict(verdict_raw)
        except ValueError:
            verdict = Verdict.FAIL

        if verdict == Verdict.PASS and relevance_score < self.PASS_THRESHOLD:
            verdict = Verdict.RETRY if retry_count < self.MAX_RETRIES else Verdict.FAIL
            reason  = f"{reason} (score {relevance_score} < threshold {self.PASS_THRESHOLD})"

        if verdict == Verdict.RETRY and not suggestions.strip():
            verdict = Verdict.FAIL
            reason  = f"{reason} (no suggestion for RETRY)"

        return CriticResult(
            verdict=verdict,
            reason=reason,
            suggestions=suggestions,
            relevance_score=relevance_score,
        )

    def should_retry(self, verdict: Verdict) -> bool:
        return verdict == Verdict.RETRY
