import asyncio
import re
import uuid
import structlog
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import json
import time
from pathlib import Path
from app.utils.reference_resolver import ReferenceResolver
from app.agents.memory.qdrant_memory import QdrantMemory
from app.utils.websocket_manager import cancellation_store
from app.orchestrator.recovery_manager import RecoveryManager
from app.db.session import get_db_context
from app.models.task import Task, Step, TaskStatus, StepStatus
from app.models.memory import TaskContext
from app.utils.websocket_manager import ws_manager
from app.agents.planner import PlannerAgent
from app.agents.executor import ExecutorAgent
from app.agents.critic import CriticAgent, CriticResult, Verdict
from app.agents.specialist.researcher_agent import ResearcherAgent
from app.agents.specialist.enginer_agent import EngineerAgent
from app.agents.specialist.writer_agent import WriterAgent
from app.orchestrator.agent_switcher import AgentSwitcher
from app.agents.memory.agent_preference_memory import AgentPreferenceMemory
from app.agents.reasoner import ReasonerAgent, ReasoningOutput
from app.agents.reflection import ReflectionAgent
from app.agents.confidence_memory import ConfidenceMemory
from app.utils.cost_tracker import global_cost_tracker
from app.agents.memory.tool_success_memory import ToolSuccessMemory
from app.agents.memory.user_feedback_memory import UserFeedbackMemory
from app.agents.memory.tool_failure_memory import ToolFailureMemory
from app.utils.llm import call_openai_with_system
from app.core.config import settings
from app.orchestrator.research import (
    ResearchPlan, ResearchSufficiencyReviewer, compact_research_evidence, normalize_research_key,
    research_stop_reason,
)

logger = structlog.get_logger()


def _tool_evidence(wrapper: str, result) -> Dict[str, Any]:
    metadata = result.metadata or {}
    status = metadata.get("validation_status") or ("success" if result.success else "failed")
    source_types = {
        "github": "repository_api", "get_weather": "weather_api",
        "news_search": "news", "semantic_scholar_search": "academic_index",
        "web_search": "web_search", "web_fetch": "web_page",
        "wikipedia_search": "encyclopedia", "python_executor": "local_computation",
    }
    return {
        "source": metadata.get("source", wrapper),
        "source_type": metadata.get("source_type", source_types.get(wrapper, "tool")),
        "wrapper": wrapper,
        "query": metadata.get("query") or metadata.get("city") or metadata.get("owner"),
        "source_url": metadata.get("source_url"),
        "retrieved_at": metadata.get("retrieved_at") or datetime.now(timezone.utc).isoformat(),
        "status": status,
        "failure_type": metadata.get("failure_type"),
        "fallback_from": metadata.get("fallback_from"),
        "provider_attempts": metadata.get("provider_attempts", []),
        "result": result.output[:8000] if result.success or status == "unverified" else "",
        "extracted_information": result.output[:8000] if result.success else "",
        "error": result.error if not result.success or status == "unverified" else None,
    }


def _format_evidence_fallback(evidence: list, research_state: Optional[dict] = None) -> str:
    """Return attributed raw findings if the synthesis model is unavailable."""
    sections = []
    for item in evidence:
        for attempt in item.get("attempts", []):
            if attempt.get("status") == "success" and attempt.get("result", "").strip():
                source = attempt.get("source") or attempt.get("wrapper") or "unknown source"
                sections.append(f"Source: {source} ({attempt.get('wrapper')})\n{attempt['result'][:1800]}")
    if not sections:
        answer = "I could not produce a verified answer because the selected information sources failed."
    else:
        answer = "I retrieved the following information, but the synthesis step was unavailable:\n\n" + "\n\n".join(sections[:6])
    failures = []
    for item in evidence:
        for attempt in item.get("attempts", []):
            if attempt.get("status") != "success":
                failures.append(f"{attempt.get('wrapper', 'source')}: {attempt.get('failure_type') or attempt.get('error') or 'failed'}")
    if failures:
        answer += "\n\nUnavailable sources: " + "; ".join(dict.fromkeys(failures))
    if research_state:
        if research_state.get("unresolved_questions"):
            answer += "\n\nUnresolved questions: " + "; ".join(research_state["unresolved_questions"])
        if research_state.get("stopping_reason"):
            answer += f"\nResearch stopped because: {research_state['stopping_reason']}"
    return answer


def _synthesis_has_grounded_links(answer: str, evidence: list) -> bool:
    """Reject synthesized citations whose URLs were not present in tool output."""
    cited = set(re.findall(r'https?://[^\s)\]>"}]+', answer or ""))
    if not cited:
        return True
    retrieved = set()
    for item in evidence:
        for attempt in item.get("attempts", []):
            if attempt.get("status") != "success":
                continue
            if attempt.get("source_url"):
                retrieved.add(str(attempt["source_url"]).rstrip(".,;"))
            for url in re.findall(r'https?://[^\s)\]>"}]+', attempt.get("result", "")):
                retrieved.add(url.rstrip(".,;"))
    return all(url.rstrip(".,;") in retrieved for url in cited)

# ---------------------------------------------------------------------------
# Keywords that indicate a user is asking about session/memory history.
# Defined once here and reused in Phase 0b, Phase 3, and session filtering.
# ---------------------------------------------------------------------------
MEMORY_QUERY_KEYWORDS = [
    "previous task",
    "previous research",
    "previous findings",
    "what did you find",
    "earlier task",
    "session history",
    "findings from previous",
    "top findings",
    "findings from the research",
    "from the research",
    "what were the findings",
    "research findings",
    "summarize previous",
    "what was the previous",
    "discussed in the previous",
    "3 bullet points",
]


def _utcnow() -> datetime:
    """Return current UTC time as a naive datetime (matches SQLAlchemy DateTime columns)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def is_memory_query(text: str) -> bool:
    """Return True if the text is asking about prior session history."""
    lower = text.lower()
    return any(k in lower for k in MEMORY_QUERY_KEYWORDS)


# Tools that are always safe to run in parallel — they are read-only,
# produce no side-effects, and don't depend on each other's output.
_PARALLEL_SAFE_TOOLS = frozenset({
    "web_search", "web_fetch", "news_search", "get_weather",
    "semantic_scholar_search", "wikipedia_search",
})


def _is_parallel_safe(instruction: str) -> bool:
    """
    Return True if a step instruction is safe to run in parallel with other
    parallel-safe steps (i.e. it only reads from the web, not files or code).
    Detection is keyword-based — conservative: false-negatives are fine,
    false-positives would be bad (running dependent steps concurrently).
    """
    low = instruction.lower()
    # Explicit tool mentions
    if any(t in low for t in _PARALLEL_SAFE_TOOLS):
        return True
    # Common planner phrasings that always imply a web lookup
    web_phrases = [
        "search for", "look up", "find information", "find the",
        "retrieve", "get current", "fetch", "what is the current",
        "latest news", "current news", "breaking news",
    ]
    return any(p in low for p in web_phrases)


def _group_parallel_steps(plan: list) -> list[list]:
    """
    Build execution levels from a DAG defined by each step's depends_on list.

    Algorithm — topological level sort:
    1. Steps with no dependencies → level 0 (run in parallel)
    2. Steps whose ALL dependencies are in completed levels → next level
    3. Repeat until all steps are assigned

    Falls back to the old keyword heuristic if no step declares depends_on
    (backwards-compatible with plans from the old planner).

    Returns a list of batches, where each batch is a list of step_data dicts
    to execute concurrently.
    """
    # Check if any step has an explicit depends_on declaration
    has_deps = any("depends_on" in s and s["depends_on"] for s in plan)
    any_declared = any("depends_on" in s for s in plan)

    if any_declared:
        # DAG path — use declared dependencies
        step_map  = {s["step"]: s for s in plan}
        completed: set = set()
        remaining = list(plan)
        batches   = []

        while remaining:
            # Steps whose dependencies are all satisfied
            ready = [
                s for s in remaining
                if all(dep in completed for dep in s.get("depends_on", []))
            ]
            if not ready:
                # Cycle or bad deps — fall back: run rest sequentially
                logger.warning("dag_cycle_detected", remaining=[s["step"] for s in remaining])
                for s in remaining:
                    batches.append([s])
                break

            batches.append(ready)
            for s in ready:
                completed.add(s["step"])
            remaining = [s for s in remaining if s["step"] not in completed]

        logger.info(
            "orchestrator_dag_batches",
            total_steps=len(plan),
            levels=len(batches),
            parallel_levels=sum(1 for b in batches if len(b) > 1),
        )
        return batches

    # Legacy path — keyword heuristic (no depends_on in plan)
    batches: list[list] = []
    current_batch: list = []

    for step_data in plan:
        if _is_parallel_safe(step_data["instruction"]):
            current_batch.append(step_data)
        else:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            batches.append([step_data])

    if current_batch:
        batches.append(current_batch)

    return batches


def classify_failure(error: Optional[str]) -> str:
    """Map a raw error string to a coarse failure category for metrics."""
    if not error:
        return "UNKNOWN"
    e = error.lower()
    if "no such file" in e:
        return "FILE_NOT_FOUND"
    if "syntaxerror" in e:
        return "SYNTAX_ERROR"
    if "command not found" in e:
        return "COMMAND_NOT_FOUND"
    return "UNKNOWN"


def export_task_trace(metrics: dict) -> None:
    """Write per-task JSON trace to the traces/ directory."""
    Path("traces").mkdir(exist_ok=True)
    with open(f"traces/task_{metrics['task_id']}.json", "w") as f:
        json.dump(metrics, f, indent=2)


def finalize_and_export(task_metrics: dict) -> None:
    """Stamp duration and flush trace to disk."""
    task_metrics["duration_sec"] = round(time.time() - task_metrics["started_at"], 2)
    export_task_trace(task_metrics)


def _fail_task(task: Task, message: str) -> None:
    """Mark a task as FAILED with a message and timestamp."""
    task.status = TaskStatus.FAILED
    task.error_message = message
    task.completed_at = _utcnow()


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------


async def execute_task_v3(task_id: str) -> None:
    """
    Autonomous agent orchestration pipeline.

    Phases
    ------
    0a. Reasoning    — understand the task, pick problem type
    0b. Coordination — route to specialist agents (researcher / engineer / writer)
    1.  Memory       — recall similar past tasks from ConfidenceMemory
    2.  Search       — decide whether web search is warranted
    3.  Planning     — break task into atomic executable steps
    4.  Execution    — run each step with critic-gated retry / recovery
    5.  Reflection   — learn from the outcome, update confidence scores
    """
    global_cost_tracker.start_task(task_id)

    task_metrics: dict = {
        "task_id": task_id,
        "started_at": time.time(),
        "total_steps": 0,
        "completed_steps": 0,
        "retries": 0,
        "failures": [],
        "step_traces": [],
        "llm_attempts": [],
        "memories_used": [],
        "created_files": [],
        "reasoning_used": False,
        "search_decision": None,
        "reflection_generated": False,
        "confidence_updates": 0,
    }

    logger.info("orchestrator_v3_started", task_id=task_id)

    # Initialise all agents outside the try block so they are guaranteed to
    # exist if an early phase crashes and Phase 4 recovery still needs them.
    reasoner = ReasonerAgent()
    planner = PlannerAgent()
    executor = ExecutorAgent()
    critic = CriticAgent()
    reflection_agent = ReflectionAgent()
    tool_success_memory = ToolSuccessMemory()
    tool_failure_memory = ToolFailureMemory()
    feedback_memory = UserFeedbackMemory()
    recovery_manager = RecoveryManager()
    agent_pref_memory = AgentPreferenceMemory()

    week4_agents = {
        "researcher": ResearcherAgent(),
        "engineer": EngineerAgent(),
        "writer": WriterAgent(),
    }
    agent_switcher = AgentSwitcher(week4_agents)
    qdrant_memory = QdrantMemory()

    reasoning_output: Optional[ReasoningOutput] = None
    reasoning_dict: Optional[dict] = None
    should_search = False

    try:
        with get_db_context() as db:

            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                logger.error("orchestrator_task_not_found", task_id=task_id)
                return

            context: Dict[str, Any] = {"task_description": task.user_input}

            # ----------------------------------------------------------------
            # Load session context if task belongs to a session
            # ----------------------------------------------------------------
            if task.session_id:
                previous_tasks = (
                    db.query(Task)
                    .filter(
                        Task.session_id == task.session_id,
                        Task.id != task_id,
                        Task.status == TaskStatus.COMPLETED,
                    )
                    .order_by(Task.created_at.desc())
                    .limit(3)
                    .all()
                )

                if previous_tasks:
                    session_context = []
                    for prev_task in reversed(previous_tasks):
                        if prev_task.status != TaskStatus.COMPLETED:
                            continue

                        # Skip tasks that were themselves memory queries
                        if is_memory_query(prev_task.user_input):
                            continue

                        prev_ctx = (
                            db.query(TaskContext)
                            .filter(TaskContext.task_id == prev_task.id)
                            .first()
                        )
                        output = ""
                        if prev_ctx:
                            output = (prev_ctx.context_data or {}).get("week4_output", "")

                        if not output:
                            completed_steps = (
                                db.query(Step)
                                .filter(
                                    Step.task_id == prev_task.id,
                                    Step.status == StepStatus.COMPLETED,
                                )
                                .order_by(Step.step_number.desc())
                                .all()
                            )
                            filtered_results = [
                                s.result[:300]
                                for s in completed_steps
                                if s.result
                                and "ENGINEERING EXECUTION" not in s.result
                                and "All agents failed" not in s.result
                            ]
                            output = "\n".join(filtered_results)

                        logger.info(
                            "session_context_debug",
                            task=prev_task.user_input,
                            output_preview=output[:300],
                        )
                        session_context.append(
                            {
                                "task": prev_task.user_input,
                                "status": prev_task.status,
                                "output": output[:500],
                                "files": prev_ctx.created_files if prev_ctx else [],
                                "entities": (
                                    prev_ctx.context_data.get("entities", [])
                                    if prev_ctx and prev_ctx.context_data
                                    else []
                                ),
                            }
                        )

                    # Only set session_history if there are non-memory entries
                    if session_context:
                        context["session_history"] = session_context
                    logger.info(
                        "session_context_loaded",
                        session_id=task.session_id,
                        num_previous_tasks=len(session_context),
                    )

            task.status = TaskStatus.RUNNING
            db.commit()

            conf_memory = ConfidenceMemory(db=db)

            task_context = TaskContext(
                id=str(uuid.uuid4()),
                task_id=task_id,
                context_data={},
                created_files=[],
                memories_used=[],
            )
            db.add(task_context)
            db.commit()

            # ================================================================
            # PHASE 0a: REASONING
            # ================================================================
            logger.info("orchestrator_reasoning_phase", task=task.user_input)
            try:
                t0 = time.time()
                reasoning_output = await reasoner.reason(task_description=task.user_input)
                reasoning_dict = reasoning_output.model_dump()

                global_cost_tracker.record_llm_call(
                    agent="reasoner",
                    model=reasoner.model,
                    response_length=len(str(reasoning_output)),
                    purpose="reasoning",
                    duration_ms=(time.time() - t0) * 1000,
                )

                task_metrics["reasoning_used"] = True
                task_metrics["reasoning_output"] = {
                    "problem_type": reasoning_output.problem_type,
                    "confidence": reasoning_output.confidence,
                    "needs_search": reasoning_output.needs_search,
                    "needs_memory": reasoning_output.needs_memory,
                }

                logger.info(
                    "orchestrator_reasoning_completed",
                    problem_type=reasoning_output.problem_type,
                    confidence=reasoning_output.confidence,
                    strategy=reasoning_output.strategy,
                )
                await ws_manager.emit(
                    task_id,
                    {
                        "phase": "reasoning",
                        "status": "completed",
                        "problem_type": reasoning_output.problem_type,
                        "confidence": reasoning_output.confidence,
                    },
                )
                if cancellation_store.is_cancelled(task_id):
                    cancellation_store.clear(task_id)
                    return
            except Exception as e:
                logger.error("orchestrator_reasoning_failed", error=str(e))

            # Capability routing is handled once by the structured planner below.
            # Specialist agents remain available to explicit recovery flows, but
            # are not invoked speculatively before the planner selects tools.
            await ws_manager.emit(task_id, {"phase": "coordination", "status": "skipped"})

            # ================================================================
            # PHASE 1: MEMORY RECALL
            # ================================================================
            similar_memories: list = []
            memory_confidence = 0.0

            if reasoning_output and reasoner.should_use_memory(reasoning_output):
                logger.info("orchestrator_memory_phase")
                try:
                    similar_memories, memory_confidence = (
                        await conf_memory.recall_with_confidence(
                            task_description=task.user_input,
                            min_confidence=0.3,
                            limit=3,
                        )
                    )
                    if similar_memories:
                        task_context.memories_used = [m["id"] for m in similar_memories]
                        task_metrics["memories_used"] = task_context.memories_used
                        task_metrics["memory_confidence"] = memory_confidence
                        db.commit()
                        logger.info(
                            "orchestrator_memories_recalled",
                            num_memories=len(similar_memories),
                            avg_confidence=memory_confidence,
                        )
                except Exception as e:
                    logger.error("orchestrator_memory_failed", error=str(e))

            # Memory recall is independent from external tool selection.
            if task.session_id:
                memories = qdrant_memory.search_memory(task.user_input, limit=2)
                # Filter out low-quality memories before injecting into context.
                junk_patterns = [
                    "ENGINEERING EXECUTION",
                    "All agents failed",
                    "Tool execution would happen here",
                    "Failed to choose appropriate tool",
                ]
                memories = [
                    m for m in memories
                    if m.get("result")
                    and not any(p in m.get("result", "") for p in junk_patterns)
                    and len(m.get("result", "").strip()) > 50
                ]
                context["qdrant_memories"] = memories
                logger.info("qdrant_memories_loaded", count=len(memories))

            # ================================================================
            # PHASE 3: PLANNING
            # ================================================================
            logger.info("orchestrator_planning")
            try:
                t0 = time.time()

                # Build base task description, optionally enriched with qdrant memories
                memory_context = ""
                if context.get("qdrant_memories"):
                    memory_context = "\n\nRELEVANT MEMORIES:\n"
                    for m in context["qdrant_memories"]:
                        memory_context += (
                            f"\nPrevious Query:\n{m.get('query', '')[:100]}\n"
                            f"Previous Result:\n{m.get('result', '')[:100]}\n"
                        )

                task_description = task.user_input + memory_context
                logger.info("planner_input", task_description=task_description[:2000])

                # If this is a memory query, inject session history into the prompt
                if context.get("session_history") and is_memory_query(task.user_input):
                    history_text = "\n".join(
                        f"Previous Task:\n{h['task']}\n\nResult Summary:\n{h['output'][:300]}"
                        for h in context["session_history"]
                    )
                    task_description = (
                        f"CURRENT TASK:\n{task.user_input}\n\n"
                        f"SESSION HISTORY (from previous tasks in this conversation):\n"
                        f"{history_text[:500]}\n\n"
                        f"INSTRUCTIONS FOR USING SESSION HISTORY:\n"
                        f"- Extract and list specific items from SESSION HISTORY\n"
                        f"- For top findings/results, parse and list them clearly\n"
                        f"- For summaries, write a proper summary of SESSION HISTORY content\n"
                        f"- Do NOT echo the raw SESSION HISTORY — extract and format\n"
                    )

                # Reference resolution: enrich query if it refers to prior entities
                if context.get("session_history"):
                    resolver = ReferenceResolver()
                    if resolver.has_reference(task.user_input):
                        resolution = resolver.resolve(
                            task.user_input, context["session_history"]
                        )
                        task_description = resolution["enriched_query"]
                        logger.info(
                            "reference_resolution_applied",
                            query=task.user_input,
                            subject=resolution["resolved_subject"],
                        )

                logger.info(
                    "planner_final_input", task_description=task_description[:2000]
                )
                planner_intent = await planner.plan_intent(task_description)
                plan = planner_intent["steps"]
                research_enabled = bool(planner_intent.get("research")) or planner_intent["mode"] == "multi_tool"
                research_started_at = time.monotonic()
                research_plan = None
                research_history = []
                if research_enabled:
                    max_initial_steps = max(1, settings.RESEARCH_MAX_TOOL_CALLS)
                    if len(plan) > max_initial_steps:
                        plan = plan[:max_initial_steps]
                    research_plan = ResearchPlan.from_intent(task.user_input, plan, planner_intent)
                    task_metrics["research"] = {
                        "plan": research_plan.model_dump(),
                        "iterations": 0,
                        "tool_calls": 0,
                        "stopping_reason": "initial_plan_pending",
                    }
                selected_tool_names = [step["tool"] for step in plan]
                should_search = any(name in {
                    "web_search", "web_fetch", "news_search", "semantic_scholar_search", "wikipedia_search"
                } for name in selected_tool_names)
                task_metrics["search_decision"] = {
                    "should_search": should_search,
                    "planner_mode": planner_intent["mode"],
                    "selected_tools": selected_tool_names,
                    "reason": "structured planner intent",
                }
                global_cost_tracker.record_llm_call(
                    agent="planner",
                    model=planner.model,
                    response_length=len(str(plan)),
                    purpose="planning",
                    duration_ms=(time.time() - t0) * 1000,
                )
            except Exception as e:
                logger.error("orchestrator_planning_failed", error=str(e))
                _fail_task(task, f"Planning failed: {e}")
                task_metrics["failures"].append(
                    {"step_number": None, "error": str(e), "category": "PLANNING_ERROR"}
                )
                db.commit()
                finalize_and_export(task_metrics)
                global_cost_tracker.complete_task(success=False)
                return

            # Direct answers and requested deliverables complete from the planner's
            # structured intent without dispatching any tool or specialist.
            if planner_intent["mode"] == "direct_response":
                final_output = planner_intent["answer"]
                direct_step = Step(
                    id=str(uuid.uuid4()), task_id=task_id, step_number=1,
                    instruction="Return the requested direct response",
                    status=StepStatus.COMPLETED, result=final_output,
                    completed_at=_utcnow(),
                )
                db.add(direct_step)
                task.status = TaskStatus.COMPLETED
                task.completed_at = _utcnow()
                task_context.context_data = {
                    **context, "planner_intent": planner_intent,
                    "final_output": final_output,
                }
                task_metrics["total_steps"] = 1
                task_metrics["completed_steps"] = 1
                db.commit()
                await ws_manager.emit(task_id, {"phase": "planning", "status": "completed", "steps": []})
                await ws_manager.emit(task_id, {"phase": "completed", "status": "completed"})
                logger.info("orchestrator_direct_response_completed", task_id=task_id)
                finalize_and_export(task_metrics)
                global_cost_tracker.complete_task(success=True)
                return

            # Persist all steps, then fetch them in one query for the execution loop
            step_numbers = []
            for step_data in plan:
                if cancellation_store.is_cancelled(task_id):
                    logger.info("orchestrator_task_cancelled", task_id=task_id)
                    cancellation_store.clear(task_id)
                    return

                db.add(
                    Step(
                        id=str(uuid.uuid4()),
                        task_id=task_id,
                        step_number=step_data["step"],
                        instruction=step_data["instruction"],
                        status=StepStatus.PENDING,
                    )
                )
                global_cost_tracker.record_step()
                step_numbers.append(step_data["step"])
            db.commit()

            # Fetch all steps at once (avoids N+1 queries)
            steps_by_number: Dict[int, Step] = {
                s.step_number: s
                for s in db.query(Step)
                .filter(
                    Step.task_id == task_id,
                    Step.step_number.in_(step_numbers),
                )
                .all()
            }

            logger.info("orchestrator_plan_created", num_steps=len(plan))
            await ws_manager.emit(
                task_id,
                {
                    "phase": "planning",
                    "status": "completed",
                    "steps": [
                        {"number": s["step"], "instruction": s["instruction"]}
                        for s in plan
                    ],
                },
            )
            if cancellation_store.is_cancelled(task_id):
                cancellation_store.clear(task_id)
                return
            task_metrics["total_steps"] = len(plan)

            # ================================================================
            # PHASE 4: EXECUTION
            # ================================================================
            context.update(
                {
                    "memories": similar_memories,
                    "should_search": should_search,
                    "planner_mode": planner_intent["mode"],
                    "preferred_tools": tool_success_memory.top_tools(),
                }
            )
            logger.info("preferred_tools_loaded", tools=context["preferred_tools"])

            # ── Group steps into parallel/sequential batches ──────────────
            step_batches = _group_parallel_steps(plan)
            # Multi-source research steps share the request-scoped SQLAlchemy
            # session and evidence context. Run them in plan order to avoid
            # concurrent session writes and partial task completion races.
            if research_enabled:
                step_batches = [[step] for step in plan]
            logger.info(
                "orchestrator_step_batches",
                total_steps=len(plan),
                total_batches=len(step_batches),
                parallel_batches=sum(1 for b in step_batches if len(b) > 1),
            )

            # ── Inner coroutine: execute one step with retry/recovery ─────
            def _research_tool_call_count() -> int:
                return sum(
                    len(context.get(f"step_{planned['step']}_evidence", []))
                    for planned in plan
                )

            async def _run_step(step_data: dict) -> bool:
                """
                Execute a single plan step. Returns True if the step succeeded
                (or was skipped), False if it hard-failed and the task should abort.
                Mutates `context`, `task_metrics`, `task_context` in the outer scope.
                """
                # Narrow types — task is guaranteed non-None here because we
                # checked it before entering Phase 4 and would have returned early.
                assert task is not None, "task must be non-None inside _run_step"

                step_number = step_data["step"]
                step = steps_by_number.get(step_number)

                if not step:
                    logger.error("orchestrator_step_not_found", step_number=step_number)
                    return True  # non-fatal — skip missing step

                # step is non-None from here — assert to satisfy type checker
                assert step is not None

                logger.info("orchestrator_executing_step", step_number=step_number)
                await ws_manager.emit(
                    task_id,
                    {
                        "phase": "step",
                        "status": "running",
                        "step_number": step_number,
                        "instruction": step.instruction,
                    },
                )

                max_retries              = min(
                    critic.MAX_RETRIES,
                    max(1, settings.RESEARCH_MAX_REPEAT_STEPS + 1),
                ) if research_enabled else critic.MAX_RETRIES
                retry_count              = 0
                step_succeeded           = False
                switched_agents_this_step: set = set()

                while retry_count < max_retries and not step_succeeded:
                    if (research_enabled
                            and _research_tool_call_count() >= settings.RESEARCH_MAX_TOOL_CALLS):
                        step.status = StepStatus.SKIPPED
                        step.error = "Research tool-call budget reached"
                        db.commit()
                        return True
                    step.status = StepStatus.RUNNING
                    step.retry_count = retry_count
                    db.commit()

                    try:
                        context["avoid_tools"] = [
                            t for t in ["python_executor", "shell_executor"]
                            if tool_failure_memory.should_avoid(t)
                        ]

                        primary_tool = step_data.get("tool", "unknown")
                        step_context = dict(context)
                        if research_enabled:
                            remaining_calls = max(1, settings.RESEARCH_MAX_TOOL_CALLS - _research_tool_call_count())
                            step_context["max_tool_attempts"] = remaining_calls
                        t0 = time.time()
                        tool_result, tool_attempts = await executor.execute_with_fallbacks(
                            instruction=step.instruction,
                            primary_tool=primary_tool,
                            fallback_tools=step_data.get("fallback_tools", []),
                            request=task.user_input,
                            context=step_context,
                        )
                        global_cost_tracker.record_llm_call(
                            agent="executor",
                            model=executor.model,
                            response_length=len(str(tool_result)),
                            purpose="execution_with_fallbacks",
                            duration_ms=(time.time() - t0) * 1000,
                        )
                        evidence_attempts = []
                        for attempted_wrapper, attempt_result in tool_attempts:
                            evidence_attempts.append(_tool_evidence(attempted_wrapper, attempt_result))
                            if attempted_wrapper in (
                                "web_search", "web_fetch", "news_search", "get_weather",
                                "semantic_scholar_search", "wikipedia_search",
                            ):
                                global_cost_tracker.record_search()

                        step.result   = tool_result.output
                        step.error    = tool_result.error
                        step.tool_name = tool_result.metadata.get("tool_name") or primary_tool
                        context[f"step_{step_number}_evidence"] = [
                            *context.get(f"step_{step_number}_evidence", []),
                            *evidence_attempts,
                        ]
                        db.commit()

                        logger.info(
                            "orchestrator_step_executed",
                            step_number=step_number,
                            success=tool_result.success,
                        )

                        if tool_result.success and (tool_result.output or "").strip() and tool_result.metadata.get("tool_name"):
                            tool_failure_memory.reset_failures(tool_result.metadata["tool_name"])

                        if tool_result.success and not (tool_result.output or "").strip():
                            tool_result.success = False
                            tool_result.error = "Tool returned an empty result"
                            tool_result.metadata["failure_type"] = "EMPTY_RESULT"

                        if not tool_result.success:
                            # Preserve the failure as task data so the final answer
                            # can report the limitation and still use other results.
                            tool_name = step.tool_name or step_data.get("tool") or "selected tool"
                            step.error = tool_result.error or "Tool returned no result"
                            step.result = f"{tool_name} failed: {step.error}"
                            step.status = StepStatus.FAILED
                            step.completed_at = _utcnow()
                            context[f"step_{step_number}_output"] = ""
                            context[f"step_{step_number}_error"] = step.result
                            context[f"step_{step_number}_success"] = False
                            context[f"step_{step_number}_tool"] = primary_tool
                            task_metrics["failures"].append({
                                "step_number": step_number,
                                "error": step.error,
                                "category": tool_result.metadata.get("failure_type", "TOOL_FAILURE"),
                            })
                            await ws_manager.emit(task_id, {
                                "phase": "step", "status": "failed",
                                "step_number": step_number, "error": step.error,
                            })
                            db.commit()
                            return True

                        t0 = time.time()
                        if research_enabled:
                            # Tool-result relevance is assessed collectively by
                            # the research reviewer after collection. Avoid a
                            # redundant per-step LLM call in this path.
                            evaluation = CriticResult(
                                verdict=Verdict.PASS,
                                reason="Nonempty wrapper result retained for research coverage review",
                                relevance_score=100,
                            )
                        else:
                            evaluation = await critic.evaluate(
                                step_instruction=step.instruction,
                                tool_result=tool_result,
                                retry_count=retry_count,
                            )
                            global_cost_tracker.record_llm_call(
                                agent="critic",
                                model=critic.model,
                                response_length=len(str(evaluation)),
                                purpose="critic",
                                duration_ms=(time.time() - t0) * 1000,
                            )

                        logger.info(
                            "orchestrator_step_evaluated",
                            step_number=step_number,
                            verdict=evaluation.verdict,
                            reason=evaluation.reason,
                        )

                        task_metrics["step_traces"].append({
                            "step_number":  step_number,
                            "attempt":      retry_count,
                            "instruction":  step.instruction,
                            "tool_success": tool_result.success,
                            "error":        tool_result.error,
                            "verdict":      evaluation.verdict.value,
                            "reason":       evaluation.reason,
                            "timestamp":    _utcnow().isoformat(),
                        })

                        # A critic disagreement should not discard successful
                        # source material in a multi-source research task. Keep
                        # the evidence and let final synthesis qualify it.
                        if (planner_intent["mode"] == "multi_tool"
                                and evaluation.verdict != Verdict.PASS):
                            critic_unavailable = evaluation.reason == "Failed to parse critic evaluation"
                            step.status = StepStatus.COMPLETED if critic_unavailable else StepStatus.FAILED
                            step.error = None if critic_unavailable else (evaluation.reason or "Critic could not validate this source")
                            step.result = tool_result.output
                            context[f"step_{step_number}_output"] = tool_result.output
                            context[f"step_{step_number}_success"] = bool(tool_result.success)
                            context[f"step_{step_number}_tool"] = step.tool_name
                            context[f"step_{step_number}_review"] = evaluation.reason
                            if not critic_unavailable:
                                task_metrics["failures"].append({
                                    "step_number": step_number,
                                    "error": step.error,
                                    "category": "REVIEW_REJECTED_EVIDENCE_RETAINED",
                                })
                            else:
                                task_metrics.setdefault("warnings", []).append({
                                    "step_number": step_number,
                                    "warning": "Critic response was unparseable; successful nonempty source evidence was retained",
                                })
                            db.commit()
                            return True

                        # ── PASS ──────────────────────────────────────────
                        if evaluation.verdict == Verdict.PASS:
                            if step.tool_name:
                                tool_success_memory.record_success(str(step.tool_name))
                            task_metrics["completed_steps"] += 1
                            step.status      = StepStatus.COMPLETED
                            step.completed_at = _utcnow()
                            step_succeeded   = True
                            feedback_memory.record_feedback(
                                query=task.user_input,
                                feedback="good",
                                answer=tool_result.output[:500],
                            )
                            await ws_manager.emit(
                                task_id,
                                {
                                    "phase": "step",
                                    "status": "completed",
                                    "step_number": step_number,
                                    "result": tool_result.output[:300],
                                },
                            )
                            context[f"step_{step_number}_output"]  = tool_result.output
                            context[f"step_{step_number}_success"] = True
                            context[f"step_{step_number}_tool"] = step.tool_name

                            filename = tool_result.metadata.get("filename")
                            if filename and filename not in (task_context.created_files or []):
                                task_context.created_files = [*(task_context.created_files or []), filename]
                                task_metrics["created_files"].append(filename)

                        # ── RETRY ─────────────────────────────────────────
                        elif evaluation.verdict == Verdict.RETRY:
                            task_metrics["retries"] += 1
                            global_cost_tracker.record_retry()
                            step.status = StepStatus.RETRYING
                            retry_count += 1
                            logger.warning(
                                "orchestrator_step_retrying",
                                step_number=step_number,
                                retry_count=retry_count,
                                suggestions=evaluation.suggestions,
                            )
                            await asyncio.sleep(1)

                        # ── FAIL ──────────────────────────────────────────
                        else:
                            if step.tool_name:
                                tool_success_memory.record_failure(str(step.tool_name))
                            logger.error("orchestrator_step_failed", step_number=step_number, reason=evaluation.reason)
                            feedback_memory.record_feedback(
                                query=task.user_input,
                                feedback="bad",
                                answer=evaluation.reason,
                            )

                            step_failure_info = {
                                "what_worked":            [],
                                "what_failed":            [evaluation.reason],
                                "root_causes":            [tool_result.error or evaluation.reason],
                                "lessons":                [],
                                "improvement_suggestions": [evaluation.suggestions or ""],
                                "pattern_quality":        0.0,
                                "confidence_updates":     {},
                            }
                            decision = recovery_manager.decide(step_failure_info)
                            logger.info("recovery_attempt", action=decision.action, reason=decision.reason)

                            if decision.action in ("retry", "retry_with_smaller_prompt"):
                                if decision.action == "retry_with_smaller_prompt":
                                    context["prompt_reduction"] = True
                                retry_count += 1
                                db.commit()
                                continue

                            elif decision.action == "switch_agent":
                                switched_result, new_agent = await agent_switcher.switch_and_execute(
                                    failed_agent="executor",
                                    instruction=step.instruction,
                                    context=context,
                                    already_tried=switched_agents_this_step,
                                )
                                if switched_result:
                                    step.result       = switched_result.output
                                    step.status       = StepStatus.COMPLETED
                                    step.completed_at = _utcnow()
                                    context[f"step_{step_number}_output"]  = switched_result.output
                                    context[f"step_{step_number}_success"] = True
                                    context["recovered_by_agent"]          = new_agent
                                    step_succeeded = True
                                    if new_agent:
                                        switched_agents_this_step.add(new_agent)
                                        agent_pref_memory.record_success(
                                            task_description=task.user_input,
                                            agent_name=new_agent,
                                        )
                                    logger.info("step_recovered_by_agent_switch", step=step_number, agent=new_agent)
                                    db.commit()
                                    break
                                else:
                                    logger.error("agent_switcher_exhausted", step=step_number, tried=list(switched_agents_this_step))

                            elif decision.action == "skip_step":
                                step.status = StepStatus.SKIPPED
                                db.commit()
                                return True   # skipped = not a hard failure

                            elif decision.action == "abort_task":
                                _fail_task(task, decision.reason)
                                db.commit()
                                if global_cost_tracker.current_task is not None:
                                    finalize_and_export(task_metrics)
                                    global_cost_tracker.complete_task(success=False)
                                return False  # signal caller to abort

                            # Hard fail — no recovery succeeded
                            step.status = StepStatus.FAILED
                            _fail_task(task, f"Step {step_number} failed: {evaluation.reason}")
                            task_metrics["failures"].append({
                                "step_number": step_number,
                                "error":       evaluation.reason,
                                "category":    classify_failure(step.error),
                            })
                            db.commit()
                            if global_cost_tracker.current_task is not None:
                                finalize_and_export(task_metrics)
                                global_cost_tracker.complete_task(success=False)
                            return False

                        db.commit()

                    except Exception as e:
                        logger.error("orchestrator_step_error", step_number=step_number, error=str(e))
                        step.error  = str(e)
                        step.status = StepStatus.FAILED
                        await ws_manager.emit(task_id, {"phase": "failed", "status": "failed", "error": str(e)})
                        _fail_task(task, f"Step {step_number} crashed: {e}")
                        task_metrics["failures"].append({
                            "step_number": step_number,
                            "error":       str(e),
                            "category":    "ORCHESTRATOR_ERROR",
                        })
                        db.commit()
                        if global_cost_tracker.current_task is not None:
                            finalize_and_export(task_metrics)
                            global_cost_tracker.complete_task(success=False)
                        return False

                if not step_succeeded:
                    logger.error("orchestrator_step_exhausted_retries", step_number=step_number)
                    step.status = StepStatus.FAILED
                    _fail_task(task, f"Step {step_number} exhausted retries")
                    task_metrics["failures"].append({
                        "step_number": step_number,
                        "error":       "Exhausted retries",
                        "category":    "RETRY_LIMIT_EXCEEDED",
                    })
                    db.commit()
                    # Only finalize if this task hasn't already been aborted
                    # by another parallel step in the same batch
                    if global_cost_tracker.current_task is not None:
                        finalize_and_export(task_metrics)
                        global_cost_tracker.complete_task(success=False)
                    return False

                return True   # step succeeded

            # ── Execute batches ───────────────────────────────────────────
            for batch in step_batches:
                if cancellation_store.is_cancelled(task_id):
                    cancellation_store.clear(task_id)
                    return

                if (research_enabled and
                        time.monotonic() - research_started_at >= settings.RESEARCH_MAX_SECONDS):
                    if research_plan:
                        research_plan.stopping_reason = "research_time_limit"
                    break

                if len(batch) == 1:
                    # Sequential step
                    ok = await _run_step(batch[0])
                    if not ok:
                        return
                else:
                    # Parallel batch — fire all steps simultaneously
                    logger.info(
                        "orchestrator_parallel_batch",
                        steps=[s["step"] for s in batch],
                        count=len(batch),
                    )
                    results = await asyncio.gather(
                        *[_run_step(s) for s in batch],
                        return_exceptions=True,
                    )
                    # If any step signalled a hard failure (False) or raised, abort
                    for r in results:
                        if isinstance(r, BaseException):
                            logger.error("orchestrator_parallel_step_exception", error=str(r))
                            _fail_task(task, f"Parallel step crashed: {r}")
                            db.commit()
                            if global_cost_tracker.current_task is not None:
                                finalize_and_export(task_metrics)
                                global_cost_tracker.complete_task(success=False)
                            return
                        if r is False:
                            return   # hard-fail already written by _run_step

            # Evaluate coverage and information gaps only for plans the planner
            # identified as complex research (or genuinely multi-source tasks).
            # Simple single-wrapper lookups retain their low-cost path.
            final_evidence: list[dict[str, Any]] = []
            if research_enabled and research_plan is not None:
                reviewer = ResearchSufficiencyReviewer(planner.model)
                seen_research_steps = {
                    normalize_research_key(item.tool, item.instruction)
                    for item in research_plan.research_steps
                }

                def _collect_research_evidence() -> list[dict[str, Any]]:
                    collected = []
                    for item in research_plan.research_steps:
                        attempts = context.get(f"step_{item.step}_evidence", [])
                        orm_step = steps_by_number.get(item.step)
                        if orm_step and orm_step.status == StepStatus.COMPLETED:
                            state = "completed"
                        elif attempts:
                            state = "failed"
                        else:
                            state = "skipped"
                        research_plan.record_step(item.step, state, item.step)
                        collected.append({
                            "evidence_id": item.step,
                            "subquestion": item.subquestion,
                            "instruction": item.instruction,
                            "tool": item.tool,
                            "status": state,
                            "attempts": attempts,
                            "result": context.get(f"step_{item.step}_output", ""),
                            "error": context.get(f"step_{item.step}_error", ""),
                        })
                    return collected

                for iteration in range(max(0, settings.RESEARCH_MAX_ITERATIONS) + 1):
                    final_evidence = _collect_research_evidence()
                    if time.monotonic() - research_started_at >= settings.RESEARCH_MAX_SECONDS:
                        research_plan.stopping_reason = "research_time_limit"
                        break
                    remaining_time = max(
                        0.1, settings.RESEARCH_MAX_SECONDS - (time.monotonic() - research_started_at)
                    )
                    assessment_started = time.monotonic()
                    try:
                        assessment = await asyncio.wait_for(
                            reviewer.assess(research_plan, final_evidence), timeout=remaining_time
                        )
                    except asyncio.TimeoutError:
                        for diagnostic in reviewer.last_call_diagnostics:
                            task_metrics["llm_attempts"].append(diagnostic)
                            global_cost_tracker.record_llm_diagnostic(diagnostic)
                        research_plan.stopping_reason = "research_time_limit_during_evidence_review"
                        break
                    for diagnostic in reviewer.last_call_diagnostics:
                        task_metrics["llm_attempts"].append(diagnostic)
                        global_cost_tracker.record_llm_diagnostic(diagnostic)
                    global_cost_tracker.record_llm_call(
                        agent="research_reviewer", model=planner.model,
                        response_length=len(str(assessment)), purpose="research_sufficiency",
                        duration_ms=(time.monotonic() - assessment_started) * 1000,
                    )
                    research_plan.iteration_count = iteration + 1
                    research_plan.unresolved_questions = assessment.unresolved_questions
                    research_plan.conflicts = assessment.conflicts
                    research_history.append({
                        "iteration": iteration + 1,
                        "sufficient": assessment.sufficient,
                        "addressed_subquestions": assessment.addressed_subquestions,
                        "unresolved_questions": assessment.unresolved_questions,
                        "conflicts": [item.model_dump() for item in assessment.conflicts],
                        "stopping_reason": assessment.stopping_reason,
                    })
                    stop_reason = research_stop_reason(
                        assessment, iteration=iteration,
                        max_iterations=max(0, settings.RESEARCH_MAX_ITERATIONS),
                        tool_calls=_research_tool_call_count(),
                        max_tool_calls=settings.RESEARCH_MAX_TOOL_CALLS,
                        elapsed_seconds=time.monotonic() - research_started_at,
                        max_seconds=settings.RESEARCH_MAX_SECONDS,
                    )
                    if stop_reason:
                        research_plan.stopping_reason = stop_reason
                        break
                    available_calls = settings.RESEARCH_MAX_TOOL_CALLS - _research_tool_call_count()
                    followups = research_plan.append_followups(
                        assessment, seen_research_steps, max_steps=available_calls
                    )
                    if not followups:
                        research_plan.stopping_reason = "no_new_valid_follow_up"
                        break
                    research_plan.iteration_count = iteration + 2
                    for followup in followups:
                        if time.monotonic() - research_started_at >= settings.RESEARCH_MAX_SECONDS:
                            research_plan.stopping_reason = "research_time_limit"
                            break
                        if _research_tool_call_count() >= settings.RESEARCH_MAX_TOOL_CALLS:
                            research_plan.stopping_reason = "maximum_tool_calls_reached"
                            break
                        number = followup["step"]
                        db_step = Step(
                            id=str(uuid.uuid4()), task_id=task_id, step_number=number,
                            instruction=followup["instruction"], status=StepStatus.PENDING,
                        )
                        db.add(db_step)
                        db.commit()
                        steps_by_number[number] = db_step
                        step_numbers.append(number)
                        plan.append(followup)
                        task_metrics["total_steps"] += 1
                        followup_state = next(
                            item for item in research_plan.research_steps if item.step == number
                        )
                        await ws_manager.emit(task_id, {
                            "phase": "research", "status": "follow_up_started",
                            "iteration": iteration + 2, "step_number": number,
                            "subquestion": followup_state.subquestion,
                        })
                        if not await _run_step(followup):
                            research_plan.stopping_reason = "follow_up_execution_failed"
                            break
                    else:
                        continue
                    break
                else:
                    research_plan.stopping_reason = "maximum_research_iterations_reached"

                final_evidence = _collect_research_evidence()
                final_evidence = [
                    {
                        **item,
                        "attempts": [
                            {**attempt, "subquestion": item["subquestion"], "evidence_id": item["evidence_id"]}
                            for attempt in item["attempts"]
                        ],
                    }
                    for item in final_evidence
                ]
                task_metrics["research"] = {
                    "plan": research_plan.model_dump(),
                    "history": research_history,
                    "iterations": research_plan.iteration_count,
                    "tool_calls": _research_tool_call_count(),
                    "stopping_reason": research_plan.stopping_reason,
                    "elapsed_seconds": round(time.monotonic() - research_started_at, 2),
                }
                context["research_state"] = task_metrics["research"]
                db.commit()

            # Synthesize tool evidence into the final answer. This also reports
            # failed wrappers explicitly instead of inventing missing facts.
            evidence = final_evidence or []
            if not evidence:
                for step_data in plan:
                    number = step_data["step"]
                    evidence.append({
                        "step": number,
                        "instruction": step_data["instruction"],
                        "attempts": context.get(f"step_{number}_evidence", []),
                        "selected_tool": context.get(f"step_{number}_tool"),
                        "success": context.get(f"step_{number}_success", False),
                        "result": context.get(f"step_{number}_output", ""),
                        "error": context.get(f"step_{number}_error", ""),
                    })
            synthesis_started = time.time()
            research_synthesis_context = ""
            if research_enabled and research_plan is not None:
                research_synthesis_context = (
                    "\n\nResearch plan and review state (include material gaps, conflicts, and stop reason):\n"
                    + json.dumps({
                        "objective": research_plan.objective,
                        "subquestions": research_plan.subquestions,
                        "completed_steps": research_plan.completed_steps,
                        "unresolved_questions": research_plan.unresolved_questions,
                        "conflicts": [item.model_dump() for item in research_plan.conflicts],
                        "iteration_count": research_plan.iteration_count,
                        "stopping_reason": research_plan.stopping_reason,
                        "assessment_history": research_history,
                    }, ensure_ascii=False, default=str)
                )
            try:
                final_output = await call_openai_with_system(
                    system_prompt=(
                        "Answer the user's request using only the tool evidence supplied. "
                        "Clearly state when a source failed and do not guess missing facts. Ignore evidence marked failed or unverified. "
                        "Never claim that a wrapper/source was attempted unless it appears in the supplied evidence. "
                        "Every URL in your answer must appear verbatim in a successful evidence result; do not invent or recall URLs. "
                        "For conflicts, describe both positions and preserve uncertainty unless the supplied research review resolves them. "
                        "Combine independent sources when the request asks for a comparison or conclusion. "
                        "Do not claim a tool succeeded unless its evidence says success. Use clean natural language, "
                        "not JSON or internal intent metadata. Name the source and note when fallback search evidence "
                        "is less authoritative than a primary API."
                    ),
                    user_prompt=(
                        f"User request:\n{task.user_input}\n\nTool evidence:\n"
                        f"{json.dumps(compact_research_evidence(evidence), ensure_ascii=False, default=str)}"
                        f"{research_synthesis_context}"
                    ),
                    model=planner.model,
                    temperature=0.1,
                    max_tokens=3000,
                    reasoning_effort="low",
                )
                if not _synthesis_has_grounded_links(final_output, evidence):
                    logger.warning("orchestrator_synthesis_rejected_unretrieved_url")
                    final_output = _format_evidence_fallback(evidence, task_metrics.get("research"))
            except Exception as exc:
                logger.warning("orchestrator_synthesis_failed", error_type=type(exc).__name__)
                final_output = _format_evidence_fallback(evidence, task_metrics.get("research"))
            global_cost_tracker.record_llm_call(
                agent="planner", model=planner.model,
                response_length=len(final_output), purpose="synthesis",
                duration_ms=(time.time() - synthesis_started) * 1000,
            )
            final_step_number = max(step_numbers, default=0) + 1
            db.add(Step(
                id=str(uuid.uuid4()), task_id=task_id,
                step_number=final_step_number,
                instruction="Synthesize the tool results into the requested answer",
                status=StepStatus.COMPLETED, result=final_output,
                completed_at=_utcnow(),
            ))
            context["final_output"] = final_output
            db.commit()

            # ================================================================
            # PHASE 5: REFLECTION & LEARNING
            # ================================================================
            task.status = TaskStatus.COMPLETED
            task.completed_at = _utcnow()
            logger.info(
                "saving_task_context",
                task=task.user_input,
                week4_output=context.get("week4_output", "")[:500],
            )
            task_context.context_data = context
            db.commit()

            # Flush batched tool success/failure stats to disk now that the task is done
            tool_success_memory.flush()

            logger.info("orchestrator_task_completed", task_id=task_id)
            await ws_manager.emit(task_id, {"phase": "completed", "status": "completed"})

            try:
                best_agent = context.get("preferred_agent") or context.get(
                    "recovered_by_agent"
                )
                if best_agent in {"researcher", "engineer", "writer"}:
                    agent_pref_memory.record_success(
                        task_description=task.user_input,
                        agent_name=best_agent,
                    )
                logger.info(
                    "agent_preference_learned",
                    task_type=(
                        reasoning_output.problem_type if reasoning_output else "general"
                    ),
                    agent=best_agent,
                )
            except Exception as e:
                logger.error("agent_preference_update_failed", error=str(e))

            try:
                t0 = time.time()
                reflection_output = await reflection_agent.reflect(
                    task=task,
                    reasoning_used=reasoning_dict,
                    search_used=should_search,
                )
                global_cost_tracker.record_llm_call(
                    agent="reflection",
                    model=reflection_agent.model,
                    response_length=len(str(reflection_output)),
                    purpose="reflection",
                    duration_ms=(time.time() - t0) * 1000,
                )

                task_metrics["reflection_generated"] = True
                task_metrics["reflection_lessons"] = reflection_output.lessons
                task_metrics["pattern_quality"] = reflection_output.pattern_quality

                logger.info(
                    "orchestrator_reflection_completed",
                    num_lessons=len(reflection_output.lessons),
                    quality=reflection_output.pattern_quality,
                )

                await conf_memory.update_confidence_from_reflection(
                    reflection=reflection_output,
                    task_pattern=(
                        reasoning_output.problem_type if reasoning_output else "general"
                    ),
                )
                task_metrics["confidence_updates"] = len(
                    reflection_output.confidence_updates
                )

                memory_id = await conf_memory.store_with_confidence(
                    pattern_type="success",
                    task_pattern=(
                        reasoning_output.problem_type if reasoning_output else "general"
                    ),
                    task_id=task.id,
                    task_description=task.user_input,
                    strategy=(
                        reflection_output.lessons[0]
                        if reflection_output.lessons
                        else "Completed successfully"
                    ),
                    tools_used=list({s.tool_name for s in task.steps if s.tool_name}),
                    steps_taken=[
                        {
                            "step": s.step_number,
                            "instruction": s.instruction,
                            "tool": s.tool_name,
                            "status": s.status,
                        }
                        for s in task.steps
                    ],
                    success=True,
                    reflection=reflection_output,
                )
                logger.info("orchestrator_learned", memory_id=memory_id)

            except Exception as e:
                logger.error("orchestrator_reflection_failed", error=str(e))

            finalize_and_export(task_metrics)
            global_cost_tracker.complete_task(success=True)
            logger.info("orchestrator_v3_completed", task_id=task_id)

    except Exception as e:
        logger.error("orchestrator_v3_error", task_id=task_id, error=str(e))
        task_metrics["failures"].append(
            {"step_number": None, "error": str(e), "category": "ORCHESTRATOR_CRASH"}
        )
        with get_db_context() as db:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                _fail_task(task, f"Orchestrator error: {e}")
                db.commit()

        finalize_and_export(task_metrics)
        global_cost_tracker.complete_task(success=False)
