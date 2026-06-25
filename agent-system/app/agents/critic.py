import structlog
from enum import Enum
from app.utils.json_parser import extract_json
from app.tools.base import ToolResult
from app.utils.llm import call_groq_with_system

logger = structlog.get_logger()


# This file answers one question after every step: "Did that work?"
# Three possible outcomes. str, Enum means these are both enum values AND strings, so they can be stored in the database as text without extra conversion.
class Verdict(str, Enum):
    PASS = "PASS"
    RETRY = "RETRY"
    FAIL = "FAIL"


# Simple container holding the verdict + why + what to try differently.
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

EVALUATION CRITERIA (in order of importance):
1. TOPICAL RELEVANCE — Is the output actually ABOUT the subject named in the
   instruction, not just sharing a keyword with it? A result that mentions
   "Python" but is about a completely different domain (e.g. astrotourism,
   epidemiology, chip architecture) is OFF-TOPIC, even if it is well-written
   and coherent. Off-topic output is a FAIL, regardless of fluency.
2. Did the tool execute without errors?
3. Is the output useful for subsequent steps?

BE STRICT ON TOPIC, LENIENT ON FORM:
- Coherence, fluency, or "the tool ran successfully" is NOT sufficient for PASS
  if the subject matter does not match the instruction. Judge the CONTENT,
  not just whether something was returned.
- Empty output may still be acceptable depending on intent, but irrelevant
  non-empty output is worse than empty output — do not reward verbosity.
- Do NOT fail a step merely because the method/process is not shown, or
  because the answer is short — as long as the CONTENT is on-topic and correct.
- If even one part of a mixed/multi-source result set is genuinely on-topic
  and useful, PASS is acceptable. If most/all results are off-topic, FAIL.

EXCEPTION — session history tasks:
- If the instruction asks to summarize, extract, or answer using SESSION HISTORY,
  verdict is PASS as long as the output is a coherent response — do NOT judge
  whether the history content matches your expectations or is factually correct.
  The executor cannot change what is in the session history; retrying is pointless.
  (This exception applies ONLY to session-history tasks, not to web search/research tasks.)

RESPOND ONLY WITH JSON.
"""

    def __init__(self, model: str = "llama-3.1-8b-instant"):
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

        try:
            metadata_str = str(tool_result.metadata)
        except Exception:
            metadata_str = "(unserializable metadata)"

        user_prompt = f"""STEP INSTRUCTION:
{step_instruction}

TOOL EXECUTION:
- Success: {tool_result.success}
- Output: {tool_result.output[:500] if tool_result.output else "(empty)"}
- Error: {tool_result.error if tool_result.error else "(none)"}
- Metadata: {metadata_str}

RETRY COUNT: {retry_count}/{self.MAX_RETRIES}

Evaluate if this step succeeded and return verdict JSON.
"""

        try:
            response = await call_groq_with_system(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self.model,
                temperature=0.1,
            )

            # ---- ROBUST JSON EXTRACTION ----
            evaluation = extract_json(response, context="critic")
            if not evaluation:
                logger.error("critic_json_error", error="No valid JSON found")
                return CriticResult(
                    verdict=(
                        Verdict.RETRY
                        if retry_count < self.MAX_RETRIES
                        else Verdict.FAIL
                    ),
                    reason="Failed to parse evaluation JSON: No valid JSON found",
                    suggestions="Ensure the response is valid JSON only",
                )

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

        except Exception as e:
            logger.error("critic_error", error=str(e))
            return CriticResult(
                verdict=(
                    Verdict.RETRY if retry_count < self.MAX_RETRIES else Verdict.FAIL
                ),
                reason=f"Evaluation error: {str(e)}",
                suggestions="",
            )

    def should_retry(self, verdict: Verdict) -> bool:
        return verdict == Verdict.RETRY
