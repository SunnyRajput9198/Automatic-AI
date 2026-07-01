import structlog
from enum import Enum

from app.utils.json_parser import extract_json
from app.tools.base import ToolResult
from app.utils.llm import call_openai_with_tools

logger = structlog.get_logger()


class Verdict(str, Enum):
    PASS = "PASS"
    RETRY = "RETRY"
    FAIL = "FAIL"


class CriticResult:
    """Result of critic evaluation."""

    def __init__(
        self,
        verdict: Verdict,
        reason: str,
        suggestions: str = "",
        relevance_score: int = 0,
    ):
        self.verdict = verdict
        self.reason = reason
        self.suggestions = suggestions
        self.relevance_score = (
            relevance_score  # 0-100, logged for future threshold tuning
        )


# ---------------------------------------------------------------------------
# Tool schema — forces structured output, no free-text JSON parsing
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

RELEVANCE SCORE GUIDE (judge TOPICAL RELEVANCE only):
- 65-100: Output is about the correct subject → PASS
- 40-64 : Partially relevant, misses the core subject → RETRY
- 0-39  : Wrong domain entirely → RETRY (first attempt) or FAIL (later)

CRITICAL RULES — READ CAREFULLY:

1. WEB SEARCH RETURNS SNIPPETS, NOT FULL DATA:
   web_search, wikipedia_search, and news_search return SHORT SNIPPETS by design.
   They cannot return live weather readings, exact census numbers, or full articles.
   If the snippet mentions the subject, that is a PASS — do not retry for missing exact values.

2. WEATHER QUERIES — ALWAYS PASS if the query mentioned temperature/weather:
   If the step instruction contains "temperature", "weather", "temp", or "climate":
   → PASS with score 70 regardless of whether the snippet has exact °C value
   → web_search and DuckDuckGo CANNOT return live weather readings — this is expected
   → Any result about the queried city = PASS

3. NEWS QUERIES — PASS if articles about the topic are returned:
   news_search for "AI news" returning articles about technology/AI companies → PASS (score 70+)
   Do NOT retry because one article in the list is off-topic — judge the majority of results.

4. FACTUAL QUERIES — PASS if any relevant snippet is returned:
   web_search for "population Delhi" returning Wikipedia about Delhi → PASS (score 70+)
   The exact number does not need to appear in the snippet.

WHAT SHOULD ACTUALLY FAIL (RETRY/FAIL):
❌ Query about AI returns results about farming, sports, entertainment → RETRY
❌ Query about Delhi returns results about a completely different city → RETRY
❌ Tool returned an error (not just incomplete data) → RETRY

KEY RULES:
- relevance_score < 65 is NOT a PASS
- Default to RETRY on first attempt for genuinely wrong-topic results
- On retry_count >= 1 with same off-topic result: use FAIL
- When RETRY: suggestions must be a concrete improved query

Call the submit_evaluation function with your assessment."""


class CriticAgent:
    """
    Evaluates step execution and decides the next action.

    Uses OpenAI tool binding to force structured output — no free-text JSON parsing.
    Threshold: relevance_score >= 65 for PASS.
    """

    MAX_RETRIES = 2
    PASS_THRESHOLD = 65  # tune upward once real score distribution is known

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
            f"- Output (first 1500 chars):\n{tool_result.output[:1500] if tool_result.output else '(empty)'}\n"
            f"- Error: {tool_result.error or '(none)'}\n"
            f"- Metadata: {metadata_str}\n\n"
            f"RETRY COUNT: {retry_count}/{self.MAX_RETRIES}\n\n"
            "Call submit_evaluation with your assessment."
        )

        try:
            # Retry up to 2 times if the model returns empty/plain text
            # instead of a tool call (model reliability issue)
            result = await call_openai_with_tools(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                tools=[_EVALUATE_TOOL],
                model=self.model,
                temperature=0.1,
                max_tokens=1000,
            )
            # If first attempt returned plain text, retry once
            if result["type"] != "tool_call":
                logger.warning(
                    "critic_tool_binding_empty_retry",
                    attempt=1,
                    preview=str(result.get("content", ""))[:100],
                )
                result = await call_openai_with_tools(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    tools=[_EVALUATE_TOOL],
                    model=self.model,
                    temperature=0.1,
                    max_tokens=1000,
                )

            if result["type"] == "tool_call" and result["name"] == "submit_evaluation":
                args = result["arguments"]
                return self._build_result(
                    verdict_raw=args.get("verdict", "FAIL"),
                    relevance_score=int(args.get("relevance_score", 0)),
                    reason=args.get("reason", "No reason provided"),
                    suggestions=args.get("suggestions", ""),
                    retry_count=retry_count,
                )

            # Fallback: model returned plain text instead of tool call
            logger.warning(
                "critic_tool_binding_fallback",
                preview=str(result.get("content", ""))[:200],
            )
            evaluation = extract_json(result.get("content", ""), context="critic")
            if evaluation:
                return self._build_result(
                    verdict_raw=evaluation.get("verdict", "FAIL"),
                    relevance_score=int(evaluation.get("relevance_score", 0)),
                    reason=evaluation.get("reason", "No reason provided"),
                    suggestions=evaluation.get("suggestions", ""),
                    retry_count=retry_count,
                )

            logger.error("critic_parse_failed")
            return CriticResult(
                verdict=(
                    Verdict.RETRY if retry_count < self.MAX_RETRIES else Verdict.FAIL
                ),
                reason="Failed to parse critic evaluation",
                suggestions="",
                relevance_score=0,
            )

        except Exception as e:
            logger.error("critic_error", error=str(e))
            return CriticResult(
                verdict=(
                    Verdict.RETRY if retry_count < self.MAX_RETRIES else Verdict.FAIL
                ),
                reason=f"Evaluation error: {str(e)}",
                suggestions="",
                relevance_score=0,
            )

    def _build_result(
        self,
        verdict_raw: str,
        relevance_score: int,
        reason: str,
        suggestions: str,
        retry_count: int,
    ) -> CriticResult:
        """
        Apply threshold and suggestion rules to produce a final CriticResult.
        Single source of truth — used by both the tool-binding path and the
        plain-text fallback path.
        """
        try:
            verdict = Verdict(verdict_raw.upper())
        except ValueError:
            verdict = Verdict.FAIL

        # Override: model said PASS but score is below threshold
        if verdict == Verdict.PASS and relevance_score < self.PASS_THRESHOLD:
            logger.warning(
                "critic_threshold_override",
                score=relevance_score,
                threshold=self.PASS_THRESHOLD,
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
            reason = f"{reason} (RETRY had no concrete suggestion; treating as FAIL)"

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
