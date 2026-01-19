
import asyncio
import uuid
import structlog
from datetime import datetime
from typing import Dict, Any
import json
import time
from pathlib import Path

from app.db.session import get_db_context
from app.models.task import Task, Step, TaskStatus, StepStatus
from app.models.memory import TaskContext
from app.agents.planner import PlannerAgent
from app.agents.executor import ExecutorAgent
from app.agents.critic import CriticAgent, Verdict

# WEEK 3 IMPORTS
from app.agents.reasoner import ReasonerAgent
from app.agents.reflection import ReflectionAgent
from app.agents.search_decider import SearchDecider
from app.agents.confidence_memory import ConfidenceMemory
from app.utils.cost_tracker import global_cost_tracker

logger = structlog.get_logger()


def classify_failure(error: str | None) -> str | None:
    """Classify failure types for better metrics"""
    if not error:
        return None
    e = error.lower()
    if "no such file" in e:
        return "FILE_NOT_FOUND"
    if "syntaxerror" in e:
        return "SYNTAX_ERROR"
    if "command not found" in e:
        return "COMMAND_NOT_FOUND"
    return "UNKNOWN"


def export_task_trace(metrics: dict):
    """Export detailed task traces"""
    Path("traces").mkdir(exist_ok=True)
    path = f"traces/task_{metrics['task_id']}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def finalize_and_export(task_metrics: dict):
    """Calculate duration and export"""
    task_metrics["duration_sec"] = round(
        time.time() - task_metrics["started_at"], 2
    )
    export_task_trace(task_metrics)


async def execute_task_v3(task_id: str):
    """
    🧠 WEEK 3 ORCHESTRATION with full cognitive pipeline.
    
    NEW FEATURES:
    - Pre-planning reasoning (ReasonerAgent)
    - Confidence-weighted memory (ConfidenceMemory)
    - Smart search decisions (SearchDecider)
    - Post-task reflection (ReflectionAgent)
    - Cost tracking throughout
    - Learning from every execution
    """
    
    # Start cost tracking
    global_cost_tracker.start_task(task_id)
    
    # Metrics tracking
    task_metrics = {
        "task_id": task_id,
        "started_at": time.time(),
        "total_steps": 0,
        "completed_steps": 0,
        "retries": 0,
        "failures": [],
        "step_traces": [],
        "memories_used": [],
        "created_files": [],
        # Week 3 metrics
        "reasoning_used": False,
        "search_decision": None,
        "reflection_generated": False,
        "confidence_updates": 0,
    }
    
    logger.info("orchestrator_v3_started", task_id=task_id)
    
    # Initialize agents
    reasoner = ReasonerAgent()
    planner = PlannerAgent()
    executor = ExecutorAgent()
    critic = CriticAgent()
    reflection_agent = ReflectionAgent()
    search_decider = SearchDecider()
    
    reasoning_output = None
    reflection_output = None
    
    try:
        with get_db_context() as db:
            # Load task
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                logger.error("orchestrator_task_not_found", task_id=task_id)
                return
            
            # Update status
            task.status = TaskStatus.RUNNING
            db.commit()
            
            # Initialize confidence memory
            conf_memory = ConfidenceMemory(db=db)
            
            # Create task context
            task_context = TaskContext(
                id=str(uuid.uuid4()),
                task_id=task_id,
                context_data={},
                created_files=[],
                memories_used=[]
            )
            db.add(task_context)
            db.commit()
            
            # ================================================================
            # PHASE 0: REASONING (NEW - Week 3 Day 1)
            # ================================================================
            logger.info("orchestrator_reasoning_phase", task=task.user_input)
            
            try:
                start_time = time.time()
                reasoning_output = await reasoner.reason(
                    task_description=task.user_input
                )
                duration_ms = (time.time() - start_time) * 1000
                
                global_cost_tracker.record_llm_call(
                    agent="reasoner",
                    model=reasoner.model,
                    response_length=len(str(reasoning_output)),
                    purpose="reasoning",
                    duration_ms=duration_ms
                )
                
                task_metrics["reasoning_used"] = True
                task_metrics["reasoning_output"] = {
                    "problem_type": reasoning_output.problem_type,
                    "confidence": reasoning_output.confidence,
                    "needs_search": reasoning_output.needs_search,
                    "needs_memory": reasoning_output.needs_memory
                }
                
                logger.info(
                    "orchestrator_reasoning_completed",
                    problem_type=reasoning_output.problem_type,
                    confidence=reasoning_output.confidence,
                    strategy=reasoning_output.strategy
                )
                
            except Exception as e:
                logger.error("orchestrator_reasoning_failed", error=str(e))
                # Continue without reasoning
            
            # ================================================================
            # PHASE 1: MEMORY RECALL with Confidence (Week 3 Day 5)
            # ================================================================
            similar_memories = []
            memory_confidence = 0.0
            
            if reasoning_output and reasoner.should_use_memory(reasoning_output):
                logger.info("orchestrator_memory_phase")
                
                try:
                    similar_memories, memory_confidence = await conf_memory.recall_with_confidence(
                        task_description=task.user_input,
                        min_confidence=0.3,
                        limit=3
                    )
                    
                    if similar_memories:
                        task_context.memories_used = [m["id"] for m in similar_memories]
                        task_metrics["memories_used"] = task_context.memories_used
                        task_metrics["memory_confidence"] = memory_confidence
                        db.commit()
                        
                        logger.info(
                            "orchestrator_memories_recalled",
                            num_memories=len(similar_memories),
                            avg_confidence=memory_confidence
                        )
                
                except Exception as e:
                    logger.error("orchestrator_memory_failed", error=str(e))
            
            # ================================================================
            # PHASE 2: SEARCH DECISION (NEW - Week 3 Day 3)
            # ================================================================
            should_search = False
            search_reason = ""
            
            if reasoning_output:
                should_search, search_reason = search_decider.should_search(
                    task_description=task.user_input,
                    reasoning=reasoning_output,
                    memory_confidence=memory_confidence if memory_confidence > 0 else None,
                    similar_memories=similar_memories
                )
                
                task_metrics["search_decision"] = {
                    "should_search": should_search,
                    "reason": search_reason
                }
                
                logger.info(
                    "orchestrator_search_decision",
                    should_search=should_search,
                    reason=search_reason
                )
            
            # ================================================================
            # PHASE 3: PLANNING with Enhanced Context
            # ================================================================
            logger.info("orchestrator_planning")
            
            try:
                start_time = time.time()
                plan = await planner.plan(task.user_input)
                duration_ms = (time.time() - start_time) * 1000
                
                global_cost_tracker.record_llm_call(
                    agent="planner",
                    model=planner.model,
                    response_length=len(str(plan)),
                    purpose="planning",
                    duration_ms=duration_ms
                )
                
            except Exception as e:
                logger.error("orchestrator_planning_failed", error=str(e))
                task.status = TaskStatus.FAILED
                task.error_message = f"Planning failed: {str(e)}"
                task_metrics["failures"].append({
                    "step_number": None,
                    "error": str(e),
                    "category": "PLANNING_ERROR",
                })
                db.commit()
                finalize_and_export(task_metrics)
                global_cost_tracker.complete_task(success=False)
                return
            
            # Create step records
            for step_data in plan:
                step = Step(
                    id=str(uuid.uuid4()),
                    task_id=task_id,
                    step_number=step_data["step"],
                    instruction=step_data["instruction"],
                    status=StepStatus.PENDING
                )
                db.add(step)
                global_cost_tracker.record_step()
            
            db.commit()
            
            logger.info("orchestrator_plan_created", num_steps=len(plan))
            task_metrics["total_steps"] = len(plan)
            
            # ================================================================
            # PHASE 4: EXECUTION with Full Context
            # ================================================================
            context: Dict[str, Any] = {
                "task_description": task.user_input,
                "memories": similar_memories,
                "reasoning": reasoning_output.dict() if reasoning_output else None,
                "should_search": should_search
            }
            
            for step_data in plan:
                step_number = step_data["step"]
                
                # Load step from DB
                step = db.query(Step).filter(
                    Step.task_id == task_id,
                    Step.step_number == step_number
                ).first()
                
                if not step:
                    logger.error("orchestrator_step_not_found", step_number=step_number)
                    continue
                
                logger.info("orchestrator_executing_step", step_number=step_number)
                
                # Retry loop
                max_retries = 3
                retry_count = 0
                step_succeeded = False
                
                while retry_count < max_retries and not step_succeeded:
                    step.status = StepStatus.RUNNING
                    step.retry_count = retry_count
                    db.commit()
                    
                    # Execute step
                    try:
                        start_time = time.time()
                        tool_result = await executor.execute_step(
                            instruction=step.instruction,
                            context=context
                        )
                        duration_ms = (time.time() - start_time) * 1000
                        
                        global_cost_tracker.record_llm_call(
                            agent="executor",
                            model=executor.model,
                            response_length=len(str(tool_result)),
                            purpose="execution",
                            duration_ms=duration_ms
                        )
                        
                        # Track search if used
                        if tool_result.metadata and tool_result.metadata.get("tool_name") == "web_search":
                            global_cost_tracker.record_search()
                        
                        # Store result
                        step.result = tool_result.output
                        step.error = tool_result.error
                        step.tool_name = tool_result.metadata.get("tool_name") if tool_result.metadata else None
                        db.commit()
                        
                        logger.info(
                            "orchestrator_step_executed",
                            step_number=step_number,
                            success=tool_result.success
                        )
                        
                        # Critic evaluation
                        start_time = time.time()
                        evaluation = await critic.evaluate(
                            step_instruction=step.instruction,
                            tool_result=tool_result,
                            retry_count=retry_count
                        )
                        duration_ms = (time.time() - start_time) * 1000
                        
                        global_cost_tracker.record_llm_call(
                            agent="critic",
                            model=critic.model,
                            response_length=len(str(evaluation)),
                            purpose="critic",
                            duration_ms=duration_ms
                        )
                        
                        logger.info(
                            "orchestrator_step_evaluated",
                            step_number=step_number,
                            verdict=evaluation.verdict,
                            reason=evaluation.reason
                        )
                        
                        # Track step trace
                        task_metrics["step_traces"].append({
                            "step_number": step_number,
                            "attempt": retry_count,
                            "instruction": step.instruction,
                            "tool_success": tool_result.success,
                            "error": tool_result.error,
                            "verdict": evaluation.verdict.value,
                            "reason": evaluation.reason,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                        
                        # Handle verdict
                        if evaluation.verdict == Verdict.PASS:
                            task_metrics["completed_steps"] += 1
                            step.status = StepStatus.COMPLETED
                            step.completed_at = datetime.utcnow()
                            step_succeeded = True
                            
                            # Add output to context
                            context[f"step_{step_number}_output"] = tool_result.output
                            context[f"step_{step_number}_success"] = True
                            
                            # Track created files
                            if tool_result.metadata.get("filename"):
                                filename = tool_result.metadata["filename"]
                                if filename not in task_context.created_files:
                                    task_context.created_files.append(filename)
                                    task_metrics["created_files"].append(filename)
                        
                        elif evaluation.verdict == Verdict.RETRY:
                            task_metrics["retries"] += 1
                            global_cost_tracker.record_retry()
                            step.status = StepStatus.RETRYING
                            retry_count += 1
                            
                            logger.warning(
                                "orchestrator_step_retrying",
                                step_number=step_number,
                                retry_count=retry_count,
                                suggestions=evaluation.suggestions
                            )
                            
                            await asyncio.sleep(1)
                        
                        else:  # FAIL
                            step.status = StepStatus.FAILED
                            step.completed_at = datetime.utcnow()
                            
                            logger.error(
                                "orchestrator_step_failed",
                                step_number=step_number,
                                reason=evaluation.reason
                            )
                            
                            # Fail entire task
                            task.status = TaskStatus.FAILED
                            task.error_message = f"Step {step_number} failed: {evaluation.reason}"
                            task.completed_at = datetime.utcnow()
                            
                            task_metrics["failures"].append({
                                "step_number": step_number,
                                "error": step.error,
                                "category": classify_failure(step.error),
                            })
                            
                            db.commit()
                            
                            # WEEK 3: Reflection on failure
                            try:
                                reflection_output = await reflection_agent.reflect(
                                    task=task,
                                    reasoning_used=reasoning_output.dict() if reasoning_output else None,
                                    search_used=should_search
                                )
                                task_metrics["reflection_generated"] = True
                                task_metrics["reflection_lessons"] = reflection_output.lessons
                                
                                # Update confidence
                                await conf_memory.update_confidence_from_reflection(
                                    reflection=reflection_output,
                                    task_pattern=reasoning_output.problem_type if reasoning_output else "unknown"
                                )
                                task_metrics["confidence_updates"] = len(reflection_output.confidence_updates)
                                
                            except Exception as e:
                                logger.error("orchestrator_reflection_failed", error=str(e))
                            
                            finalize_and_export(task_metrics)
                            global_cost_tracker.complete_task(success=False)
                            return
                        
                        db.commit()
                    
                    except Exception as e:
                        logger.error(
                            "orchestrator_step_error",
                            step_number=step_number,
                            error=str(e)
                        )
                        
                        step.error = str(e)
                        step.status = StepStatus.FAILED
                        db.commit()
                        
                        task.status = TaskStatus.FAILED
                        task.error_message = f"Step {step_number} crashed: {str(e)}"
                        task.completed_at = datetime.utcnow()
                        
                        task_metrics["failures"].append({
                            "step_number": step_number,
                            "error": str(e),
                            "category": "ORCHESTRATOR_ERROR",
                        })
                        
                        db.commit()
                        finalize_and_export(task_metrics)
                        global_cost_tracker.complete_task(success=False)
                        return
                
                # Check retry exhaustion
                if not step_succeeded:
                    logger.error(
                        "orchestrator_step_exhausted_retries",
                        step_number=step_number
                    )
                    
                    step.status = StepStatus.FAILED
                    task.status = TaskStatus.FAILED
                    task.error_message = f"Step {step_number} exhausted retries"
                    task.completed_at = datetime.utcnow()
                    
                    task_metrics["failures"].append({
                        "step_number": step_number,
                        "error": "Exhausted retries",
                        "category": "RETRY_LIMIT_EXCEEDED",
                    })
                    
                    db.commit()
                    finalize_and_export(task_metrics)
                    global_cost_tracker.complete_task(success=False)
                    return
            
            # ================================================================
            # PHASE 5: SUCCESS - REFLECTION & LEARNING (NEW - Week 3 Day 4)
            # ================================================================
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            task_context.context_data = context
            db.commit()
            
            logger.info("orchestrator_task_completed", task_id=task_id)
            
            # Generate reflection
            try:
                start_time = time.time()
                reflection_output = await reflection_agent.reflect(
                    task=task,
                    reasoning_used=reasoning_output.dict() if reasoning_output else None,
                    search_used=should_search
                )
                duration_ms = (time.time() - start_time) * 1000
                
                global_cost_tracker.record_llm_call(
                    agent="reflection",
                    model=reflection_agent.model,
                    response_length=len(str(reflection_output)),
                    purpose="reflection",
                    duration_ms=duration_ms
                )
                
                task_metrics["reflection_generated"] = True
                task_metrics["reflection_lessons"] = reflection_output.lessons
                task_metrics["pattern_quality"] = reflection_output.pattern_quality
                
                logger.info(
                    "orchestrator_reflection_completed",
                    num_lessons=len(reflection_output.lessons),
                    quality=reflection_output.pattern_quality
                )
                
                # Update confidence scores
                await conf_memory.update_confidence_from_reflection(
                    reflection=reflection_output,
                    task_pattern=reasoning_output.problem_type if reasoning_output else "general"
                )
                task_metrics["confidence_updates"] = len(reflection_output.confidence_updates)
                
                # Store in memory with confidence
                memory_id = await conf_memory.store_with_confidence(
                    pattern_type="success",
                    task_pattern=reasoning_output.problem_type if reasoning_output else "general",
                    task_id=task.id,
                    task_description=task.user_input,
                    strategy=reflection_output.lessons[0] if reflection_output.lessons else "Completed successfully",
                    tools_used=list(set(step.tool_name for step in task.steps if step.tool_name)),
                    steps_taken=[{
                        "step": s.step_number,
                        "instruction": s.instruction,
                        "tool": s.tool_name,
                        "status": s.status
                    } for s in task.steps],
                    success=True,
                    reflection=reflection_output
                )
                
                logger.info("orchestrator_learned", memory_id=memory_id)
                
            except Exception as e:
                logger.error("orchestrator_reflection_failed", error=str(e))
            
            finalize_and_export(task_metrics)
            global_cost_tracker.complete_task(success=True)
            
            logger.info("orchestrator_v3_completed", task_id=task_id)
    
    except Exception as e:
        logger.error("orchestrator_v3_error", task_id=task_id, error=str(e))
        
        task_metrics["failures"].append({
            "step_number": None,
            "error": str(e),
            "category": "ORCHESTRATOR_CRASH",
        })
        
        with get_db_context() as db:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = TaskStatus.FAILED
                task.error_message = f"Orchestrator error: {str(e)}"
                task.completed_at = datetime.utcnow()
                db.commit()
        
        finalize_and_export(task_metrics)
        global_cost_tracker.complete_task(success=False)