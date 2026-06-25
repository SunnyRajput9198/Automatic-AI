import importlib.util as _ilu
import sys
import structlog
import app.tools
import app.tools.base
import app.tools.web_search
import app
import app.utils
import app.utils.json_parser
import app.utils.llm
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, START, END

logger = structlog.get_logger()


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

ResearcherAgent = _ra.ResearcherAgent
CriticAgent     = _critic.CriticAgent
Verdict         = _critic.Verdict


# ─────────────────────────────────────────────────────────────────────────────
# Shared state schema
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    task:           str
    session_id:     Optional[str]

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

_researcher = ResearcherAgent()
_critic_agent = CriticAgent()


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────

async def researcher_node(state: AgentState) -> AgentState:
    """Run ResearcherAgent, populate search fields."""
    logger.info("graph_researcher_node",
                task=state["task"][:80],
                retry=state.get("retry_count", 0))

    result = await _researcher.execute(task=state["task"])


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
    from app.tools.base import ToolResult as _TR
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
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing function (3-way)
# ─────────────────────────────────────────────────────────────────────────────

MAX_RETRIES = 2

def route_after_critic(state: AgentState) -> str:
    """
    PASS  → end
    RETRY → researcher (if retry_count < MAX_RETRIES)
    FAIL  → end
    """
    verdict     = state.get("critic_verdict", "FAIL")
    retry_count = state.get("retry_count", 0)

    if verdict == Verdict.PASS:
        logger.info("graph_routing", decision="end (PASS)")
        return "end"

    if verdict == Verdict.RETRY and retry_count < MAX_RETRIES:
        logger.info("graph_routing",
                    decision="retry researcher",
                    attempt=retry_count + 1)
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
    g.add_node("researcher",     researcher_node)
    g.add_node("critic",         critic_node)
    g.add_node("increment_retry", increment_retry)

    # Edges
    g.add_edge(START,              "researcher")
    g.add_edge("researcher",       "critic")
    g.add_edge("increment_retry",  "researcher")   # retry loop

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
