import importlib.util as _ilu
import re
import sys
import time
import structlog
import asyncio
from app.tools.base import ToolResult as _TR
from langchain_core.runnables import RunnableConfig
from typing import cast, Literal
from app.utils.llm import call_openai_with_system
from app.utils.file_manager import FileManager
from app.agents.critic import CriticAgent,Verdict
from app.agents.planner import PlannerAgent
from app.agents.specialist.researcher_agent import ResearcherAgent
from app.core.config import settings
from typing import TypedDict, Optional, List, cast
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = structlog.get_logger()

RESEARCHER_TIMEOUT_SEC = 35
TASK_TIMEOUT_SEC       = 90          # hard cap across all retries
# ─────────────────────────────────────────────────────────────────────────────
# Checkpointer factory
# ─────────────────────────────────────────────────────────────────────────────
async def get_checkpointer() -> AsyncPostgresSaver:
    """Not used directly — see run_research_graph which manages the context."""
    raise NotImplementedError("Use run_research_graph directly")

class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    task:           str
    session_id:     Optional[str]
    started_at:     float            

    # ── Planner outputs ────────────────────────────────────────────────────
    plan_steps:     List[str]        # all step instructions from planner
    planned_query:  str              # first step instruction (used by researcher)

    # ── Researcher outputs ─────────────────────────────────────────────────
    search_query:   str
    search_results: str
    num_results:    int
    sources_used:   str

    # ── Critic fields ──────────────────────────────────────────────────────
    critic_verdict:    Literal["PASS","RETRY","FAIL",""]      
    critic_reason:      str
    critic_suggestions: str
    retry_count:        int

    # ── Final ──────────────────────────────────────────────────────────────
    final_output:      str
    synthesized_output: str   # LLM-processed clean answer
    saved_file:        str    # filename if written to workspace
    success:           bool
    confidence:        float
    errors:            List[str]


# ─────────────────────────────────────────────────────────────────────────────
# Agent singletons
# ─────────────────────────────────────────────────────────────────────────────

planner      = PlannerAgent()
researcher   = ResearcherAgent()
critic_agent = CriticAgent()

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
        # Priority: extract quoted query from patterns like:
        #   'use web_search with query "X"'
        #   'web_search with query "X" and ...'
        quoted_query = re.search(
            r'(?:web_search\s+(?:with\s+)?query|query\s*=)\s*["\']([^"\']{5,})["\']',
            instruction, re.IGNORECASE
        )
        if quoted_query:
            return quoted_query.group(1).strip()

        cleaned = _PLANNER_PREFIXES.sub("", instruction).strip()
        # Strip any residual leading preposition/noise (e.g. "about X" → "X")
        cleaned = _DANGLING_PREP.sub("", cleaned).strip()
        # Strip trailing instructions like "and save..." or "and return..."
        cleaned = re.sub(
            r"\s+and\s+(save|return|store|write|output|append).*$",
            "", cleaned, flags=re.IGNORECASE
        ).strip()
        # Remove trailing punctuation left over
        cleaned = re.sub(r"[.,:;]+$", "", cleaned).strip()
        return cleaned if len(cleaned) >= 5 else instruction

    try:
        steps = await planner.plan(state["task"])
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
            researcher.execute(task=effective_task),
            timeout=RESEARCHER_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.error("graph_researcher_timeout",
                    task=effective_task[:80],
                    timeout=RESEARCHER_TIMEOUT_SEC)
        return {
            **state,
            "search_query":      effective_task,
            "search_results":    "",
            "num_results":       0,
            "sources_used":      "",
            "final_output":      "",
            "synthesized_output": "",
            "saved_file":        "",
            "success":           False,
            "confidence":        0.0,
            "errors": state.get("errors", []) + [f"Researcher timed out after {RESEARCHER_TIMEOUT_SEC}s"],
        }

    return {
        **state,
        "search_query":      result.metadata.get("query", ""),
        "search_results":    result.output,
        "num_results":       result.metadata.get("num_results", 0),
        "sources_used":      result.metadata.get("source", ""),
        "final_output":      result.output,
        "synthesized_output": "",
        "saved_file":        "",
        "success":           result.success,
        "confidence":        result.confidence,
        "errors":            result.errors,
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

    evaluation = await critic_agent.evaluate(
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


SYNTHESIZER_SYSTEM_PROMPT = """You are a research synthesizer. You receive raw web search results and produce a clean, well-structured answer.

Your job:
1. Read the raw search results carefully
2. If SESSION CONTEXT is provided, USE IT as the primary source — it contains research already done in this conversation
3. Extract the most relevant and accurate information
4. Write a clear, concise answer to the original research question
5. Cite sources inline where possible (e.g. "According to Wikipedia, ...")
6. Use bullet points or sections when the topic has multiple distinct aspects

If SESSION CONTEXT is present and relevant:
- Draw from it first before the raw search results
- Synthesize both sources when the question builds on prior research
- Example: if session has "vector database research" and question is "which is best" → use the session data to compare

DO NOT:
- Repeat the raw search result format
- Include result numbers or source labels from the raw output
- Make up information not present in the results or context

Output a clean prose answer (with optional bullet points/headers). No JSON."""


async def synthesizer_node(state: AgentState) -> AgentState:
    """
    LLM post-processing node: turns raw search results into a clean answer
    and saves it to the workspace as a .txt file.
    """
    raw = state.get("search_results", "")
    task = state["task"]

    if not raw.strip():
        return {**state, "synthesized_output": "", "saved_file": ""}

    logger.info("graph_synthesizer_node", task=task[:60])

    # Separate session context from the bare task if it was injected
    session_marker = "SESSION CONTEXT (use this to answer, do not search for it again):"
    if session_marker in task:
        parts = task.split(session_marker, 1)
        bare_task = parts[0].strip()
        session_ctx_text = parts[1].strip() if len(parts) > 1 else ""
        user_prompt = (
            f"RESEARCH QUESTION:\n{bare_task}\n\n"
            f"SESSION CONTEXT (prior research in this conversation):\n{session_ctx_text}\n\n"
            f"NEW SEARCH RESULTS:\n{raw[:2000]}\n\n"
            f"Write a clean, structured answer using both the session context and new search results."
        )
    else:
        user_prompt = (
            f"RESEARCH QUESTION:\n{task}\n\n"
            f"RAW SEARCH RESULTS:\n{raw[:3000]}\n\n"
            f"Write a clean, structured answer to the research question."
        )

    try:
        synthesized = await call_openai_with_system(
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
            max_tokens=1500,
        )
    except Exception as e:
        logger.error("graph_synthesizer_error", error=str(e))
        synthesized = raw  # fall back to raw if LLM fails

    # Save to workspace file
    saved_file = ""
    file_confirmation = ""
    try:
        fm = FileManager(base_dir=settings.WORKSPACE_DIR)
        # Derive a clean filename from the task (max 40 chars, snake_case)
        slug = re.sub(r"[^a-z0-9]+", "_", task.lower())[:40].strip("_")
        filename = f"research_{slug}.txt"
        content = f"Research Question: {task}\n\n{synthesized}"
        success = fm.write_file(filename, content)
        if success:
            saved_file = filename
            file_confirmation = f"\n\nwrote {len(content)} characters to {filename}"
            logger.info("graph_synthesizer_saved", filename=filename)
        else:
            logger.warning("graph_synthesizer_save_failed", filename=filename)
    except Exception as e:
        logger.error("graph_synthesizer_save_error", error=str(e))

    return {
        **state,
        "synthesized_output": synthesized,
        "saved_file":         saved_file,
        "final_output":       synthesized + file_confirmation,  # frontend regex picks up filename
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing function (3-way)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
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


def _build_graph_definition() -> StateGraph:
    """Return an uncompiled StateGraph (checkpointer is attached at compile time)."""
    g = StateGraph(AgentState)

    g.add_node("planner",         planner_node)
    g.add_node("researcher",      researcher_node)
    g.add_node("critic",          critic_node)
    g.add_node("increment_retry", increment_retry)
    g.add_node("synthesizer",     synthesizer_node)

    g.add_edge(START,             "planner")
    g.add_edge("planner",         "researcher")
    g.add_edge("researcher",      "critic")
    g.add_edge("increment_retry", "researcher")
    g.add_edge("synthesizer",     END)

    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end":   "synthesizer",
            "retry": "increment_retry",
        },
    )

    return g


# ── Compiled singleton (no checkpointer — used when checkpointing is off) ────
# loop_v3 calls run_research_graph() which attaches the checkpointer per call.
_graph_definition = _build_graph_definition()
research_graph    = _graph_definition.compile()   # fallback / test usage without checkpointer

# ─────────────────────────────────────────────────────────────────────────────
# Checkpointed invocation helper — used by loop_v3
# ─────────────────────────────────────────────────────────────────────────────
async def run_research_graph(
    initial_state: AgentState,
    thread_id: str,
) -> AgentState:
    """
    Invoke the research graph with a PostgresSaver checkpointer.
    The async context manager is held open for the duration of the graph run
    so the connection stays alive across all node transitions.
    """
    conn_string = settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

    try:
        async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
            await checkpointer.setup()
            compiled = _graph_definition.compile(checkpointer=checkpointer)
            config   = RunnableConfig(configurable={"thread_id": thread_id})

            logger.info("graph_invoking_with_checkpoint", thread_id=thread_id)
            result = cast(AgentState, await compiled.ainvoke(initial_state, config=config))
            logger.info("graph_completed_with_checkpoint",
                        thread_id=thread_id, verdict=result.get("critic_verdict"))
            return result

    except Exception as e:
        logger.warning(
            "graph_checkpointer_unavailable",
            error=str(e),
            fallback="running without checkpointer",
        )
        return cast(AgentState, await research_graph.ainvoke(initial_state))



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
        synthesized_output="",
        saved_file="",
        success=False,
        confidence=0.0,
        errors=[],
    )
# if __name__ == "__main__":
#     # Save visualization as PNG
#     png_bytes = research_graph.get_graph().draw_mermaid_png()
#     with open("graph.png", "wb") as f:
#         f.write(png_bytes)

#     print("Graph visualization saved as graph.png")