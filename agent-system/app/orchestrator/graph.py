import importlib.util as _ilu
import re
import sys
import time
import structlog
import app.tools
import app.tools.base
import app.tools.web_search
import asyncio
import app
import app.utils
import app.utils.json_parser
import app.utils.llm
from app.tools.base import ToolResult as _TR
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END

logger = structlog.get_logger()

RESEARCHER_TIMEOUT_SEC = 35
TASK_TIMEOUT_SEC       = 90          # hard cap across all retries

def _load(path: str, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = _ilu.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{name}' from '{path}'")
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load only the files we actually need — skip agents/__init__.py entirely
_base   = _load("app/agents/base_agent.py",                          "app.agents.base_agent")
_ra     = _load("app/agents/specialist/researcher_agent.py",         "app.agents.specialist.researcher_agent")
_critic = _load("app/agents/critic.py",                              "app.agents.critic")
_plan   = _load("app/agents/planner.py",                             "app.agents.planner")

ResearcherAgent = _ra.ResearcherAgent
CriticAgent     = _critic.CriticAgent
Verdict         = _critic.Verdict
PlannerAgent    = _plan.PlannerAgent


# ─────────────────────────────────────────────────────────────────────────────
# Shared state schema
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    task:           str
    session_id:     Optional[str]
    started_at:     float            # unix timestamp — for total time cap

    # ── Planner outputs ────────────────────────────────────────────────────
    plan_steps:     List[str]        # all step instructions from planner
    planned_query:  str              # first step instruction (used by researcher)

    # ── Researcher outputs ─────────────────────────────────────────────────
    search_query:   str
    search_results: str
    num_results:    int
    sources_used:   str

    # ── Critic fields ──────────────────────────────────────────────────────
    critic_verdict:     str          # "PASS" | "RETRY" | "FAIL"
    critic_reason:      str
    critic_suggestions: str
    retry_count:        int

    # ── Final ──────────────────────────────────────────────────────────────
    final_output: str
    success:      bool
    confidence:   float
    errors:       List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Agent singletons
# ─────────────────────────────────────────────────────────────────────────────

_planner      = PlannerAgent()
_researcher   = ResearcherAgent()
_critic_agent = CriticAgent()


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

async def planner_node(state: AgentState) -> AgentState:
    """
    Decompose the task into steps.
    Extracts the search subject from the first step instruction and stores
    it as planned_query — stripped of tool-directive prefixes so the
    researcher gets a clean search term, not "Use web_search to find X".
    """
    logger.info("graph_planner_node", task=state["task"][:80])

    # Prefixes the planner commonly emits that we want to strip
    _PLANNER_PREFIXES = re.compile(
        r"^(use\s+web_search\s+to\s+(find|search\s+for|look\s+up|get|retrieve|search)?\s*"
        r"|search\s+(the\s+web\s+)?for\s+"
        r"|find\s+information\s+(about|on)\s+"
        r"|find\s+"
        r"|look\s+up\s+"
        r"|research\s+"
        r"|retrieve\s+"
        r"|get\s+information\s+(about|on)\s+)",
        re.IGNORECASE,
    )
    # Residual dangling words/prepositions left after prefix stripping
    _DANGLING_PREP = re.compile(
        r"^(information\s+(about|on)\s+|about|on|for|regarding|concerning|related\s+to)\s*",
        re.IGNORECASE,
    )

    def _clean_instruction(instruction: str) -> str:
        """Strip planner tool-directive prefix to get the bare search subject."""
        cleaned = _PLANNER_PREFIXES.sub("", instruction).strip()
        # Strip any residual leading preposition/noise (e.g. "about X" → "X")
        cleaned = _DANGLING_PREP.sub("", cleaned).strip()
        # Remove trailing punctuation left over
        cleaned = re.sub(r"[.,:;]+$", "", cleaned).strip()
        return cleaned if len(cleaned) >= 5 else instruction

    try:
        steps = await _planner.plan(state["task"])
        instructions = [s["instruction"] for s in steps]

        # Use cleaned first instruction as the search query
        raw_first     = instructions[0] if instructions else state["task"]
        planned_query = _clean_instruction(raw_first)

        logger.info("graph_planner_done",
                    num_steps=len(instructions),
                    raw_first=raw_first[:80],
                    planned_query=planned_query[:80])

        return {
            **state,
            "plan_steps":    instructions,
            "planned_query": planned_query,
        }

    except Exception as e:
        # Planner failure is non-fatal — fall back to raw task
        logger.error("graph_planner_error", error=str(e))
        return {
            **state,
            "plan_steps":    [state["task"]],
            "planned_query": state["task"],
        }

def _build_effective_task(state: AgentState) -> str:
    """
    Priority order for what the researcher actually searches for:
    1. On retry: critic_suggestions "query: X" pattern
    2. On first attempt: planned_query (from planner)
    3. Fallback: raw task
    """
    retry_count = state.get("retry_count", 0)
    suggestions = state.get("critic_suggestions", "") or ""

    # On retry, use critic's suggested query if available
    if retry_count > 0 and suggestions.strip():
        # Pattern 1: explicit "query: X" or "try query: X"
        match = re.search(r"(?:try\s+)?query:\s*(.+)", suggestions, flags=re.IGNORECASE)
        if match:
            new_query = match.group(1).strip().strip('"').strip("'")
            if new_query:
                logger.info("graph_researcher_retry_query",
                            original=state["task"][:60],
                            new_query=new_query)
                return new_query

        # Pattern 2: single-quoted or double-quoted term anywhere in suggestion
        quoted = re.findall(r"['\"]([^'\"]{5,})['\"]", suggestions)
        if quoted:
            new_query = quoted[0].strip()
            logger.info("graph_researcher_retry_query",
                        original=state["task"][:60],
                        new_query=new_query)
            return new_query

        # Pattern 3: "such as X" or "like X" — take everything after it
        such_as = re.search(r"(?:such\s+as|like)\s+(.+)", suggestions, flags=re.IGNORECASE)
        if such_as:
            new_query = re.sub(r"[.,:;]+$", "", such_as.group(1).strip()).strip()
            if len(new_query) >= 5:
                logger.info("graph_researcher_retry_query",
                            original=state["task"][:60],
                            new_query=new_query)
                return new_query

        # Fallback: the suggestion looks like pure advisory prose — ignore it
        # and fall through to planned_query so we don't send garbage to search.
        logger.info("graph_researcher_retry_fallback", suggestions=suggestions[:60])
        # intentional fall-through to planned_query / raw task

    # First attempt — prefer planner's decomposed instruction
    planned = state.get("planned_query", "").strip()
    if planned:
        return planned

    return state["task"]


async def researcher_node(state: AgentState) -> AgentState:
    """Run ResearcherAgent, populate search fields."""
    effective_task = _build_effective_task(state)

    logger.info("graph_researcher_node",
                task=effective_task[:80],
                retry=state.get("retry_count", 0))

    try:
        result = await asyncio.wait_for(
            _researcher.execute(task=effective_task),
            timeout=RESEARCHER_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.error("graph_researcher_timeout",
                    task=effective_task[:80],
                    timeout=RESEARCHER_TIMEOUT_SEC)
        return {
            **state,
            "search_query": effective_task,
            "search_results": "",
            "num_results": 0,
            "sources_used": "",
            "final_output": "",
            "success": False,
            "confidence": 0.0,
            "errors": state.get("errors", []) + [f"Researcher timed out after {RESEARCHER_TIMEOUT_SEC}s"],
        }

    return {
        **state,
        "search_query":   result.metadata.get("query", ""),
        "search_results": result.output,
        "num_results":    result.metadata.get("num_results", 0),
        "sources_used":   result.metadata.get("source", ""),
        "final_output":   result.output,
        "success":        result.success,
        "confidence":     result.confidence,
        "errors":         result.errors,
    }

async def critic_node(state: AgentState) -> AgentState:
    """
    Evaluate researcher output.
    Builds a ToolResult-like object from state so CriticAgent can score it.
    """
    logger.info("graph_critic_node",
                num_results=state.get("num_results", 0),
                retry=state.get("retry_count", 0))

    # Build a ToolResult from the researcher's output so CriticAgent works
    # without modification (it expects a ToolResult).
    tool_result = _TR(
        success=state["success"],
        output=state["search_results"],
        error=state["errors"][0] if state["errors"] else None,
        metadata={
            "tool_name":  "web_search",
            "num_results": state["num_results"],
            "source":      state["sources_used"],
        },
    )

    evaluation = await _critic_agent.evaluate(
        step_instruction=state["task"],
        tool_result=tool_result,
        retry_count=state.get("retry_count", 0),
    )

    logger.info("graph_critic_verdict",
                verdict=evaluation.verdict.value,
                reason=evaluation.reason[:100])

    return {
        **state,
        "critic_verdict":     evaluation.verdict.value,
        "critic_reason":      evaluation.reason,
        "critic_suggestions": evaluation.suggestions,
        "confidence":         {"PASS": 0.9, "RETRY": 0.5, "FAIL": 0.1}.get(
                                evaluation.verdict.value, 0.3
                            ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing function (3-way)
# ─────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 2

def route_after_critic(state: AgentState) -> str:
    """
    PASS  → end
    RETRY → researcher (if retry_count < MAX_RETRIES AND task not timed out)
    FAIL  → end, UNLESS this is the very first attempt (retry_count == 0),
            in which case give it one reformulated retry before giving up —
            but only if the total task time cap hasn't been exceeded.
    """
    verdict     = state.get("critic_verdict", "FAIL")
    retry_count = state.get("retry_count", 0)

    # Hard time cap: if the task has been running too long, stop regardless
    started_at = state.get("started_at", 0.0)
    if started_at and (time.time() - started_at) >= TASK_TIMEOUT_SEC:
        logger.info("graph_routing", decision="end (task timed out)")
        return "end"

    if verdict == Verdict.PASS:
        logger.info("graph_routing", decision="end (PASS)")
        return "end"

    if verdict == Verdict.RETRY and retry_count < MAX_RETRIES:
        logger.info("graph_routing", decision="retry researcher", attempt=retry_count + 1)
        return "retry"

    # Fix C: don't give up on the very first attempt, even if critic said FAIL —
    # give it one free reformulated retry before surrendering.
    if verdict == Verdict.FAIL and retry_count == 0:
        suggestions = state.get("critic_suggestions", "") or ""
        if suggestions.strip():
            logger.info("graph_routing",
                        decision="retry researcher (FAIL override at attempt 0)",
                        attempt=1)
            return "retry"
        # No suggestion — still retry with a generic reformulation
        logger.info("graph_routing",
                    decision="retry researcher (FAIL override, generic reformulation)",
                    attempt=1)
        return "retry"

    logger.info("graph_routing", decision="end (FAIL/exhausted)")
    return "end"


def increment_retry(state: AgentState) -> AgentState:
    """Bump retry_count before looping back to researcher."""
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_researcher_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("planner",         planner_node)
    g.add_node("researcher",      researcher_node)
    g.add_node("critic",          critic_node)
    g.add_node("increment_retry", increment_retry)

    # Edges
    g.add_edge(START,             "planner")
    g.add_edge("planner",         "researcher")
    g.add_edge("researcher",      "critic")
    g.add_edge(START,              "planner")      # 👈 START ab planner pe jaata hai
    g.add_edge("planner",          "researcher")   # 👈 planner -> researcher
    g.add_edge("increment_retry", "researcher")    # retry loop

    # Conditional: critic → end OR retry
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end":   END,
            "retry": "increment_retry",
        },
    )

    return g.compile()


# ── Compiled singleton ────────────────────────────────────────────────────────
research_graph = build_researcher_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def make_initial_state(task: str, session_id: Optional[str] = None) -> AgentState:
    return AgentState(
        task=task,
        session_id=session_id,
        started_at=time.time(),
        plan_steps=[],
        planned_query="",
        search_query="",
        search_results="",
        num_results=0,
        sources_used="",
        critic_verdict="",
        critic_reason="",
        critic_suggestions="",
        retry_count=0,
        final_output="",
        success=False,
        confidence=0.0,
        errors=[],
    )