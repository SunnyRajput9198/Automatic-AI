import asyncio
import uuid
from langgraph.func import task
import structlog
from datetime import datetime
from typing import Any, Dict, Optional
import json
import time
from app.utils.file_manager import FileManager
from pathlib import Path
from app.utils.websocket_manager import cancellation_store
from app.orchestrator.recovery_manager import RecoveryManager
from app.db.session import get_db_context
from app.models.task import Task, Step, TaskStatus, StepStatus
from app.models.memory import TaskContext
from app.utils.websocket_manager import ws_manager
from app.agents.planner import PlannerAgent
from app.agents.executor import ExecutorAgent
from app.agents.critic import CriticAgent, Verdict
from app.agents.coordinator.coordinator_agent import CoordinatorAgent
from app.agents.specialist.researcher_agent import ResearcherAgent
from app.agents.specialist.enginer_agent import EngineerAgent
from app.agents.specialist.writer_agent import WriterAgent
from app.orchestrator.agent_switcher import AgentSwitcher
from app.agents.memory.agent_preference_memory import AgentPreferenceMemory
from app.agents.reasoner import ReasonerAgent, ReasoningOutput
from app.agents.reflection import ReflectionAgent
from app.agents.search_decider import SearchDecider
from app.agents.confidence_memory import ConfidenceMemory
from app.utils.cost_tracker import global_cost_tracker
from app.agents.memory.tool_failure_memory import ToolFailureMemory

logger = structlog.get_logger()


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


# creates task/task_id.json file in traces/ directory with all the metrics for that task, which can be used for analysis and debugging
def export_task_trace(metrics: dict) -> None:
    """Write per-task JSON trace to the traces/ directory."""
    Path("traces").mkdir(exist_ok=True)
    with open(f"traces/task_{metrics['task_id']}.json", "w") as f:
        json.dump(metrics, f, indent=2)


def finalize_and_export(task_metrics: dict) -> None:
    """Stamp duration and flush trace to disk."""
    task_metrics["duration_sec"] = round(time.time() - task_metrics["started_at"], 2)
    export_task_trace(task_metrics)


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
    tool_failure_memory = ToolFailureMemory()
    search_decider = SearchDecider()
    recovery_manager = RecoveryManager()
    agent_pref_memory = AgentPreferenceMemory()

    week4_agents = {
        "researcher": ResearcherAgent(),
        "engineer": EngineerAgent(),
        "writer": WriterAgent(),
    }
    coordinator = CoordinatorAgent(week4_agents)
    agent_switcher = AgentSwitcher(week4_agents)

    # FIX: extract reasoning_dict once so we never call .model_dump() 5×
    # on the same object, and so the None-guard is in one place.
    reasoning_output: Optional[ReasoningOutput] = None
    reasoning_dict: Optional[dict] = None  # populated after Phase 0a succeeds
    should_search = False

    try:
        with get_db_context() as db:

            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                logger.error("orchestrator_task_not_found", task_id=task_id)
                return

            context: Dict[str, Any] = {"task_description": task.user_input}

            # Load session context if task belongs to a session
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
                        # Only learn from successful tasks
                        if prev_task.status != TaskStatus.COMPLETED:
                            continue

                        # Skip memory/session-history questions
                        task_text = prev_task.user_input.lower()

                        memory_keywords = [
                            "previous task",
                            "session history",
                            "what did you find",
                            "summarize previous",
                            "findings from previous",
                            "what was the previous",
                            "previous research",
                            "discussed in the previous",
                            "3 bullet points",
                            "earlier task",
                        ]

                        if any(k in task_text for k in memory_keywords):
                            continue

                        prev_ctx = (
                            db.query(TaskContext)
                            .filter(TaskContext.task_id == prev_task.id)
                            .first()
                        )
                        output = ""
                        if prev_ctx:
                            context_data = prev_ctx.context_data or {}
                            output = context_data.get("week4_output", "")

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

                            filtered_results = []

                            for s in completed_steps:
                                if not s.result:
                                    continue

                                if "ENGINEERING EXECUTION" in s.result:
                                    continue

                                if "All agents failed" in s.result:
                                    continue

                                filtered_results.append(s.result[:300])

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
                            }
                        )

                    context["session_history"] = session_context
                    logger.info(
                        "session_context_loaded",
                        session_id=task.session_id,
                        num_previous_tasks=len(previous_tasks),
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
                reasoning_output = await reasoner.reason(
                    task_description=task.user_input
                )

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

            # ================================================================
            # PHASE 0b: MULTI-AGENT COORDINATION
            # Kept separate from 0a so a coordination crash doesn't swallow
            # the reasoning-failed log message.
            # ================================================================
            try:
                preferred_agent = agent_pref_memory.get_preferred_agent(task.user_input)
                if preferred_agent:
                    context["preferred_agent"] = preferred_agent
                    logger.info("preferred_agent_applied", agent=preferred_agent)

                memory_keywords = [
    "previous task",
    "previous research",
    "what did you find",
    "earlier task",
    "session history",
    "findings from previous",
    "top findings",
    "findings from the research",
    "from the research",
    "previous findings",
    "what were the findings",
    "research findings",
]

                is_memory_query = any(
                    k in task.user_input.lower() for k in memory_keywords
                )

                if not is_memory_query:
                    logger.info("orchestrator_coordination_phase")

                    coordination_result = await coordinator.coordinate(
                        task.user_input, context=context
                    )
                    logger.info(
                        "coordination_result_debug",
                        success=coordination_result.success,
                        successful_agents=coordination_result.successful_agents,
                        final_output=coordination_result.final_output[:500],
                    )
                    context.update(
                        {
                            "reasoning": reasoning_dict,
                            "week4_output": coordination_result.final_output,
                        }
                    )

                    task_metrics["week4_agents_used"] = coordination_result.total_agents
                    task_metrics["week4_successful_agents"] = (
                        coordination_result.successful_agents
                    )
                    await ws_manager.emit(
                        task_id,
                        {
                            "phase": "coordination",
                            "status": "completed",
                            "agents_used": coordination_result.total_agents,
                        },
                    )

                else:
                    logger.info(
                        "skipping_coordination_for_memory_query",
                        task=task.user_input,
                    )
                    await ws_manager.emit(
                        task_id,
                        {
                            "phase": "coordination",
                            "status": "completed",
                        },
                    )
                if cancellation_store.is_cancelled(task_id):
                    cancellation_store.clear(task_id)
                    return
            except Exception as e:
                logger.error("orchestrator_coordination_failed", error=str(e))

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
                        # Reassign (not append) so SQLAlchemy detects the change
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

            # ================================================================
            # PHASE 2: SEARCH DECISION
            # ================================================================
            if reasoning_output:
                should_search, search_reason = search_decider.should_search(
                    task_description=task.user_input,
                    reasoning=reasoning_output,
                    memory_confidence=(
                        memory_confidence if memory_confidence > 0 else None
                    ),
                    similar_memories=similar_memories,
                )
                task_metrics["search_decision"] = {
                    "should_search": should_search,
                    "reason": search_reason,
                }
                logger.info(
                    "orchestrator_search_decision",
                    should_search=should_search,
                    reason=search_reason,
                )
            # ================================================================
            # PHASE 3: PLANNING
            # ================================================================
            logger.info("orchestrator_planning")
            try:
                t0 = time.time()
                # Build task description with session context
                task_description = task.user_input
                logger.info("planner_input", task_description=task_description[:2000])
                memory_keywords = [
                    "previous task",
                    "previous research",
                    "what did you find",
                    "earlier task",
                    "session history",
                    "findings from previous",
                ]

                is_memory_query = any(
                    k in task.user_input.lower() for k in memory_keywords
                )

                if context.get("session_history") and is_memory_query:
                    history_text = "\n".join([f"""
                    Previous Task:
                    {h['task']}

                    Result Summary:
                    {h['output'][:300]}
                    """ for h in context["session_history"]])
                    history_text_truncated = history_text[:500]  # limit context size
                    task_description = f"""
CURRENT TASK:
{task.user_input}

SESSION HISTORY (from previous tasks in this conversation):
{history_text_truncated}

INSTRUCTIONS FOR USING SESSION HISTORY:
- If the user asks about previous findings, extract and list specific items from SESSION HISTORY
- If the user asks for top findings/results, parse the SESSION HISTORY output and list them clearly
- If the user asks to summarize, write a proper summary of SESSION HISTORY content
- Use python_executor to process and format the SESSION HISTORY into a clean answer
- Do NOT just echo the raw SESSION HISTORY — extract and format the relevant information
"""

                plan = await planner.plan(task_description)
                global_cost_tracker.record_llm_call(
                    agent="planner",
                    model=planner.model,
                    response_length=len(str(plan)),
                    purpose="planning",
                    duration_ms=(time.time() - t0) * 1000,
                )
            except Exception as e:
                logger.error("orchestrator_planning_failed", error=str(e))
                task.status = TaskStatus.FAILED
                task.error_message = f"Planning failed: {e}"
                task.completed_at = datetime.utcnow()
                task_metrics["failures"].append(
                    {"step_number": None, "error": str(e), "category": "PLANNING_ERROR"}
                )
                db.commit()
                finalize_and_export(task_metrics)
                global_cost_tracker.complete_task(success=False)
                return

            # Persist all steps, then fetch them in one query for the execution loop
            step_numbers = []
            for step_data in plan:
                # Check cancellation before each step
                if cancellation_store.is_cancelled(task_id):
                    logger.info("orchestrator_task_cancelled", task_id=task_id)
                    cancellation_store.clear(task_id)
                    return

                step_number = step_data["step"]
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

            # FIX: fetch all steps at once instead of one query per step (N+1)
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
                {"memories": similar_memories, "should_search": should_search}
            )

            for step_data in plan:
                step_number = step_data["step"]
                step = steps_by_number.get(step_number)

                if not step:
                    logger.error("orchestrator_step_not_found", step_number=step_number)
                    continue

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

                max_retries = 2
                retry_count = 0
                step_succeeded = False

                while retry_count < max_retries and not step_succeeded:
                    step.status = StepStatus.RUNNING
                    step.retry_count = retry_count
                    db.commit()

                    try:
                        # Build avoid list from persistent failure memory
                        context["avoid_tools"] = [
                            t
                            for t in ["python_executor", "shell_executor"]
                            if tool_failure_memory.should_avoid(t)
                        ]

                        t0 = time.time()
                        tool_result = await executor.execute_step(
                            instruction=step.instruction, context=context
                        )
                        global_cost_tracker.record_llm_call(
                            agent="executor",
                            model=executor.model,
                            response_length=len(str(tool_result)),
                            purpose="execution",
                            duration_ms=(time.time() - t0) * 1000,
                        )

                        if tool_result.metadata.get("tool_name") == "web_search":
                            global_cost_tracker.record_search()

                        step.result = tool_result.output
                        step.error = tool_result.error
                        step.tool_name = tool_result.metadata.get("tool_name")
                        db.commit()

                        logger.info(
                            "orchestrator_step_executed",
                            step_number=step_number,
                            success=tool_result.success,
                        )
                        if tool_result.success and tool_result.metadata.get(
                            "tool_name"
                        ):
                            tool_failure_memory.reset_failures(
                                tool_result.metadata["tool_name"]
                            )

                        t0 = time.time()
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

                        task_metrics["step_traces"].append(
                            {
                                "step_number": step_number,
                                "attempt": retry_count,
                                "instruction": step.instruction,
                                "tool_success": tool_result.success,
                                "error": tool_result.error,
                                "verdict": evaluation.verdict.value,
                                "reason": evaluation.reason,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

                        # ── PASS ──────────────────────────────────────────
                        if evaluation.verdict == Verdict.PASS:
                            task_metrics["completed_steps"] += 1
                            step.status = StepStatus.COMPLETED
                            step.completed_at = datetime.utcnow()
                            step_succeeded = True
                            await ws_manager.emit(
                                task_id,
                                {
                                    "phase": "step",
                                    "status": "completed",
                                    "step_number": step_number,
                                    "result": tool_result.output[:300],
                                },
                            )

                            context[f"step_{step_number}_output"] = tool_result.output
                            context[f"step_{step_number}_success"] = True

                            filename = tool_result.metadata.get("filename")
                            if filename and filename not in task_context.created_files:
                                # FIX: reassign so SQLAlchemy detects the JSON column change.
                                # In-place .append() is silently ignored by the ORM.
                                task_context.created_files = [
                                    *(task_context.created_files or []),
                                    filename,
                                ]
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
                            logger.error(
                                "orchestrator_step_failed",
                                step_number=step_number,
                                reason=evaluation.reason,
                            )

                            reflection_output = None
                            try:
                                reflection_output = await reflection_agent.reflect(
                                    task=task,
                                    reasoning_used=reasoning_dict,  # already a dict or None
                                    search_used=should_search,
                                )
                            except Exception as e:
                                logger.error("reflection_failed", error=str(e))

                            if reflection_output:
                                decision = recovery_manager.decide(
                                    reflection_output.model_dump()
                                )
                                logger.info(
                                    "recovery_attempt",
                                    action=decision.action,
                                    reason=decision.reason,
                                )

                                if decision.action in (
                                    "retry",
                                    "retry_with_smaller_prompt",
                                ):
                                    if decision.action == "retry_with_smaller_prompt":
                                        context["prompt_reduction"] = True
                                    retry_count += 1
                                    db.commit()
                                    continue

                                elif decision.action == "switch_agent":
                                    switched_result, new_agent = (
                                        await agent_switcher.switch_and_execute(
                                            failed_agent="executor",
                                            instruction=step.instruction,
                                            context=context,
                                        )
                                    )
                                    if switched_result:
                                        step.result = switched_result.output
                                        step.status = StepStatus.COMPLETED
                                        step.completed_at = datetime.utcnow()
                                        context[f"step_{step_number}_output"] = (
                                            switched_result.output
                                        )
                                        context[f"step_{step_number}_success"] = True
                                        context["recovered_by_agent"] = new_agent
                                        step_succeeded = True
                                        if new_agent:
                                            agent_pref_memory.record_success(
                                                task_description=task.user_input,
                                                agent_name=new_agent,
                                            )
                                        logger.info(
                                            "step_recovered_by_agent_switch",
                                            step=step_number,
                                            agent=new_agent,
                                        )
                                        db.commit()
                                        break

                                elif decision.action == "skip_step":
                                    step.status = StepStatus.SKIPPED
                                    db.commit()
                                    break

                                elif decision.action == "abort_task":
                                    task.status = TaskStatus.FAILED
                                    task.error_message = decision.reason
                                    task.completed_at = datetime.utcnow()
                                    db.commit()
                                    finalize_and_export(task_metrics)
                                    global_cost_tracker.complete_task(success=False)
                                    return

                            # Hard fail — no recovery succeeded
                            step.status = StepStatus.FAILED
                            task.status = TaskStatus.FAILED
                            task.error_message = (
                                f"Step {step_number} failed: {evaluation.reason}"
                            )
                            task.completed_at = datetime.utcnow()
                            task_metrics["failures"].append(
                                {
                                    "step_number": step_number,
                                    "error": evaluation.reason,
                                    "category": classify_failure(step.error),
                                }
                            )
                            db.commit()
                            finalize_and_export(task_metrics)
                            global_cost_tracker.complete_task(success=False)
                            return

                        db.commit()

                    except Exception as e:
                        logger.error(
                            "orchestrator_step_error",
                            step_number=step_number,
                            error=str(e),
                        )
                        step.error = str(e)
                        step.status = StepStatus.FAILED
                        task.status = TaskStatus.FAILED
                        await ws_manager.emit(
                            task_id,
                            {"phase": "failed", "status": "failed", "error": str(e)},
                        )
                        task.error_message = f"Step {step_number} crashed: {e}"
                        task.completed_at = datetime.utcnow()
                        task_metrics["failures"].append(
                            {
                                "step_number": step_number,
                                "error": str(e),
                                "category": "ORCHESTRATOR_ERROR",
                            }
                        )
                        db.commit()
                        finalize_and_export(task_metrics)
                        global_cost_tracker.complete_task(success=False)
                        return

                if not step_succeeded:
                    logger.error(
                        "orchestrator_step_exhausted_retries", step_number=step_number
                    )
                    step.status = StepStatus.FAILED
                    task.status = TaskStatus.FAILED
                    task.error_message = f"Step {step_number} exhausted retries"
                    task.completed_at = datetime.utcnow()
                    task_metrics["failures"].append(
                        {
                            "step_number": step_number,
                            "error": "Exhausted retries",
                            "category": "RETRY_LIMIT_EXCEEDED",
                        }
                    )
                    db.commit()
                    finalize_and_export(task_metrics)
                    global_cost_tracker.complete_task(success=False)
                    return

            # ================================================================
            # PHASE 5: REFLECTION & LEARNING
            # ================================================================
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            logger.info(
                "saving_task_context",
                task=task.user_input,
                week4_output=context.get("week4_output", "")[:500],
            )
            task_context.context_data = context
            db.commit()

            logger.info("orchestrator_task_completed", task_id=task_id)
            await ws_manager.emit(
                task_id, {"phase": "completed", "status": "completed"}
            )

            try:
                best_agent = context.get("preferred_agent") or context.get("recovered_by_agent")

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
                    reasoning_used=reasoning_dict,  # dict or None — no .model_dump() needed
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
                task.status = TaskStatus.FAILED
                task.error_message = f"Orchestrator error: {e}"
                task.completed_at = datetime.utcnow()
                db.commit()

        finalize_and_export(task_metrics)
        global_cost_tracker.complete_task(success=False)
