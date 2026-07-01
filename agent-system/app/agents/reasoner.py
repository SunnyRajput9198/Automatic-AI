import json
import structlog
from typing import Dict, List, Optional

from pydantic import BaseModel
from app.utils.json_parser import extract_json
from app.utils.llm import call_llm_with_system

logger = structlog.get_logger()


class ReasoningOutput(BaseModel):
    """Structured reasoning output from the ReasonerAgent."""

    problem_type: str  # "file_operation" | "web_research" | "calculation" |
    # "data_transformation" | "system_operation" | "mixed"
    strategy: str  # High-level approach in 1-2 sentences
    needs_memory: bool  # Should we check past experiences?
    needs_search: bool  # Should we search the web?
    likely_tools: List[str]  # Expected tools to use
    uncertainties: List[str]  # What could go wrong?
    confidence: float  # 0.0 – 1.0


class ReasonerAgent:
    """
    Pre-planning reasoning agent.

    Runs BEFORE the Planner to give the orchestrator a strategic view of the
    task: what kind of problem it is, what tools are likely needed, and how
    confident we are in the analysis.
    """

    SYSTEM_PROMPT = """You are a strategic reasoning agent. Your job is to analyze a task BEFORE planning begins.

Your analysis helps the system:
1. Choose the right approach
2. Avoid known failure patterns
3. Use resources efficiently (memory, search, tools)
4. Set realistic expectations

ANALYSIS FRAMEWORK:

Problem Types:
- file_operation: Reading, writing, managing files
- web_research: Searching for information online
- calculation: Math, data processing, algorithms
- data_transformation: Parse, convert, format data
- system_operation: Shell commands, system queries
- mixed: Combination of above

Strategy Guidelines:
- For file operations: Check if file exists first, use file_* tools
- For web research: Use web_search, may need multiple searches
- For calculations: Use python_executor with clear logic
- For unknown topics: Definitely need web search
- For repeated tasks: Check memory for past solutions

Tool Predictions:
- file_read, file_write, file_list, file_delete: File operations
- web_search, web_fetch: Internet research
- python_executor: Calculations, data processing
- shell_executor: System commands

Confidence Assessment:
- 0.9-1.0: Very clear task, standard approach
- 0.7-0.9: Clear task, minor uncertainties
- 0.5-0.7: Some ambiguity, multiple approaches possible
- 0.3-0.5: Significant uncertainty, need experimentation
- 0.0-0.3: Very unclear, high risk of failure

RESPONSE FORMAT (JSON only):
{
    "problem_type": "file_operation|web_research|calculation|data_transformation|system_operation|mixed",
    "strategy": "High-level approach in 1-2 sentences",
    "needs_memory": true|false,
    "needs_search": true|false,
    "likely_tools": ["tool1", "tool2"],
    "uncertainties": ["uncertainty1", "uncertainty2"],
    "confidence": 0.85
}

EXAMPLES:

Task: "Create a file called test.txt with hello world"
{
    "problem_type": "file_operation",
    "strategy": "Use file_write to create new file with specified content",
    "needs_memory": false,
    "needs_search": false,
    "likely_tools": ["file_write"],
    "uncertainties": ["file may already exist"],
    "confidence": 0.95
}

Task: "Search for the latest developments in quantum computing"
{
    "problem_type": "web_research",
    "strategy": "Use web_search to find recent articles, may need multiple searches for depth",
    "needs_memory": false,
    "needs_search": true,
    "likely_tools": ["web_search", "web_fetch"],
    "uncertainties": ["topic is rapidly evolving", "need to verify recency of sources"],
    "confidence": 0.75
}

Task: "Calculate fibonacci(100)"
{
    "problem_type": "calculation",
    "strategy": "Use python_executor with iterative approach to avoid recursion limits",
    "needs_memory": true,
    "needs_search": false,
    "likely_tools": ["python_executor"],
    "uncertainties": ["large number may need special handling"],
    "confidence": 0.9
}

RESPOND ONLY WITH JSON."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model

    async def reason(
        self,
        task_description: str,
        past_memories: Optional[List[Dict]] = None,
    ) -> ReasoningOutput:
        """
        Analyze task and produce a ReasoningOutput.

        Args:
            task_description: The user's task string.
            past_memories:    Optional list of similar past task dicts from ConfidenceMemory.

        Returns:
            ReasoningOutput with strategic analysis, or a safe conservative
            default if the LLM call or JSON parsing fails.
        """
        logger.info("reasoner_starting", task=task_description)

        memory_context = ""
        if past_memories:
            memory_context = (
                f"\n\nPAST SIMILAR TASKS:\n{json.dumps(past_memories, indent=2)}"
            )

        user_prompt = (
            f"TASK TO ANALYZE:\n{task_description}{memory_context}\n\n"
            "Analyze this task strategically. Consider:\n"
            "- What type of problem is this?\n"
            "- What approach would work best?\n"
            "- Do we need to check past experiences (memory)?\n"
            "- Do we need to search the web?\n"
            "- What tools will likely be needed?\n"
            "- What could go wrong?\n"
            "- How confident are you in this assessment?\n\n"
            "Return JSON only."
        )

        try:
            response = await call_llm_with_system(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self.model,
                temperature=0.2,
            )

            reasoning_data = extract_json(response, context="reasoner")
            if not reasoning_data:
                logger.error(
                    "reasoner_json_error",
                    error="No valid JSON found",
                    response=response[:500],
                )
                return _safe_default(
                    "Unable to analyze — proceeding with caution", "JSON parse error"
                )

            reasoning = ReasoningOutput(
                problem_type=reasoning_data.get("problem_type", "mixed"),
                strategy=reasoning_data.get("strategy", ""),
                needs_memory=bool(reasoning_data.get("needs_memory", False)),
                needs_search=bool(reasoning_data.get("needs_search", False)),
                likely_tools=reasoning_data.get("likely_tools", []),
                uncertainties=reasoning_data.get("uncertainties", []),
                confidence=float(reasoning_data.get("confidence", 0.5)),
            )

            logger.info(
                "reasoner_completed",
                problem_type=reasoning.problem_type,
                confidence=reasoning.confidence,
                needs_search=reasoning.needs_search,
                needs_memory=reasoning.needs_memory,
            )

            return reasoning

        except Exception as e:
            logger.error("reasoner_error", error=str(e))
            return _safe_default("Error in analysis — using safe defaults", str(e))

    # ------------------------------------------------------------------
    # Convenience helpers used by loop_v3
    # ------------------------------------------------------------------

    def should_use_memory(self, reasoning: ReasoningOutput) -> bool:
        """
        Return True if a memory lookup is warranted.
        Triggers when the reasoner asked for it, or when confidence
        is low enough that past experience might help.
        """
        return reasoning.needs_memory or reasoning.confidence < 0.7


def _safe_default(strategy: str, uncertainty: str) -> ReasoningOutput:
    """Return a conservative ReasoningOutput when analysis fails."""
    return ReasoningOutput(
        problem_type="mixed",
        strategy=strategy,
        needs_memory=True,
        needs_search=True,
        likely_tools=[],
        uncertainties=[uncertainty],
        confidence=0.2,
    )
