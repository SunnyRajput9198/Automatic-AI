import json
import structlog
from typing import Dict, Any
from enum import Enum

from app.utils.llm import call_llm
from app.tools.base import ToolResult

logger = structlog.get_logger()


class Verdict(str, Enum):
    PASS = "PASS"
    RETRY = "RETRY"
    FAIL = "FAIL"


class CriticResult:
    """Result of critic evaluation"""

    def __init__(self, verdict: Verdict, reason: str, suggestions: str = ""):
        self.verdict = verdict
        self.reason = reason
        self.suggestions = suggestions


class CriticAgent:
    """
    Evaluates step execution and decides next action.

    THIS IS WHAT MAKES THE SYSTEM AUTONOMOUS.
    """

    MAX_RETRIES = 2

    SYSTEM_PROMPT = """You are a critical evaluator agent. Your job is to:
1. Analyze if a step execution was successful
2. Decide if retry is needed
3. Provide suggestions for improvement

VERDICT OPTIONS:
- PASS: Step completed successfully, continue to next step
- RETRY: Step failed but can be retried with modifications
- FAIL: Step failed and cannot be recovered

RESPONSE FORMAT (JSON only):
{
  "verdict": "PASS|RETRY|FAIL",
  "reason": "detailed explanation of why",
  "suggestions": "specific changes to try (only for RETRY)"
}

EVALUATION CRITERIA:
- Did the tool execute without errors?
- Does the output match the step's intent?
- Is the output useful for subsequent steps?

BE STRICT BUT FAIR:
- Empty output may still be success
- Error messages don't always mean failure
- Judge based on intent, not verbosity

RESPOND ONLY WITH JSON.
"""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        # Claude is excellent for critic / judgment tasks
        self.model = model

    async def evaluate(
        self,
        step_instruction: str,
        tool_result: ToolResult,
        retry_count: int = 0,
    ) -> CriticResult:

        logger.info(
            "critic_evaluating",
            instruction=step_instruction,
            success=tool_result.success,
            retry_count=retry_count,
        )

        # Hard stop: exceeded retries
        if retry_count >= self.MAX_RETRIES:
            return CriticResult(
                verdict=Verdict.FAIL,
                reason=f"Step exceeded maximum retries ({self.MAX_RETRIES})",
                suggestions="",
            )

        # Fail fast if tool keeps failing
        if not tool_result.success and retry_count >= self.MAX_RETRIES - 1:
            return CriticResult(
                verdict=Verdict.FAIL,
                reason="Tool failed repeatedly and is no longer retryable",
                suggestions="",
            )

        user_prompt = f"""STEP INSTRUCTION:
{step_instruction}

TOOL EXECUTION:
- Success: {tool_result.success}
- Output: {tool_result.output[:300] if tool_result.output else "(empty)"}
- Error: {tool_result.error if tool_result.error else "(none)"}
- Metadata: {json.dumps(tool_result.metadata)}

RETRY COUNT: {retry_count}/{self.MAX_RETRIES}

Evaluate if this step succeeded and return verdict JSON.
"""

        try:
            response = await call_llm(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.1,
            )

            # ---- ROBUST JSON EXTRACTION ----
            response_text = response.strip()
            start = response_text.find("{")
            end = response_text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise json.JSONDecodeError(
                    "No valid JSON object found", response_text, 0
                )

            evaluation = json.loads(response_text[start : end + 1])

            verdict_raw = evaluation.get("verdict", "FAIL")
            try:
                verdict = Verdict(verdict_raw.upper())
            except ValueError:
                verdict = Verdict.FAIL

            reason = evaluation.get("reason", "No reason provided")
            suggestions = evaluation.get("suggestions", "")

            logger.info(
                "critic_evaluated",
                verdict=verdict,
                reason=reason,
            )

            return CriticResult(
                verdict=verdict,
                reason=reason,
                suggestions=suggestions,
            )

        except json.JSONDecodeError as e:
            logger.error("critic_json_error", error=str(e))
            return CriticResult(
                verdict=Verdict.RETRY
                if retry_count < self.MAX_RETRIES
                else Verdict.FAIL,
                reason=f"Failed to parse evaluation JSON: {str(e)}",
                suggestions="Ensure the response is valid JSON only",
            )

        except Exception as e:
            logger.error("critic_error", error=str(e))
            return CriticResult(
                verdict=Verdict.RETRY
                if retry_count < self.MAX_RETRIES
                else Verdict.FAIL,
                reason=f"Evaluation error: {str(e)}",
                suggestions="",
            )

    def should_retry(self, verdict: Verdict) -> bool:
        return verdict == Verdict.RETRY
