import structlog
from enum import Enum
from app.utils.json_parser import extract_json
from app.tools.base import ToolResult
from app.utils.llm import call_openai_with_system

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
- RETRY: Step failed but a different approach (e.g. a reformulated search
    query) could plausibly fix it on a second attempt
- FAIL: Step failed and no reasonable retry would help
RESPONSE FORMAT (JSON only):
{
    "verdict": "PASS|RETRY|FAIL",
    "relevance_score": <0-100 integer, how well the output matches the instruction's subject>,
    "reason": "detailed explanation of why",
    "suggestions": "specific changes to try (REQUIRED for RETRY)"
}

SCORING GUIDE:
- 80-100: Output is clearly about the exact subject named in the instruction → PASS
- 40-79: Partially relevant, mentions related concepts but misses the core subject → RETRY
- 0-39: Completely different domain/topic → RETRY (first attempt) or FAIL (later attempts)

EVALUATION CRITERIA (in order of importance):
1. TOPICAL RELEVANCE — Is the output actually ABOUT the subject named in the
    instruction, not just sharing a keyword with it? A result that mentions
    "Python" but is about a completely different domain (e.g. astrotourism,
    epidemiology, chip architecture) is OFF-TOPIC, even if it is well-written
    and coherent. Off-topic output is NOT a PASS, regardless of fluency.
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
    and useful, PASS is acceptable. If most/all results are off-topic, do not
    default to FAIL — first check whether RETRY applies (see below).

WHEN TO USE RETRY (for off-topic / wrong-result web search steps):
- DEFAULT TO RETRY for off-topic results on the FIRST attempt (retry_count=0),
    even if you are not fully confident a new query will help. A reformulated
    query is almost always worth one try before giving up — FAIL on the first
    attempt should be rare.
- If the output is off-topic because the SEARCH QUERY was likely too broad,
    too narrow, ambiguous, or used the wrong terminology — and a more specific
    or differently-worded query could plausibly retrieve the right subject —
    use RETRY, not FAIL.
- When you choose RETRY, "suggestions" MUST contain a concrete improved
    search query (e.g. "try query: FastAPI vs Django benchmarks async Python"),
    not vague advice like "use better keywords".
- Only use FAIL on the first attempt if the topic is fundamentally unsearchable
    (e.g. asks about private/internal information, or something that doesn't
    exist) — not merely because the current result happened to be off-topic.
- On later attempts (retry_count >= 1), if a reformulated query STILL returns
    the same or a similarly off-topic result, FAIL is appropriate — repeating
    near-identical results is a sign that retrying further will not help.
RESPOND ONLY WITH JSON.
"""

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
            response = await call_openai_with_system(
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

            # Safety net: if the model said RETRY but forgot to give a
            # concrete suggestion, downgrade to FAIL rather than retrying blindly.
            if verdict == Verdict.RETRY and not suggestions.strip():
                logger.error(
                    "critic_retry_missing_suggestions",
                    reason=reason,
                )
                verdict = Verdict.FAIL
                reason = f"{reason} (RETRY requested without a concrete suggestion; treating as FAIL)"

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
