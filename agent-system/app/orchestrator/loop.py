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
from app.agents.memory import MemoryAgent

logger = structlog.get_logger()

def classify_failure(error: str | None) -> str | None:
    """YOUR ENHANCEMENT: Classify failure types for better metrics"""
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
    """YOUR ENHANCEMENT: Export detailed task traces"""
    Path("traces").mkdir(exist_ok=True)
    path = f"traces/task_{metrics['task_id']}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
        

def finalize_and_export(task_metrics: dict):
    """YOUR ENHANCEMENT: Calculate duration and export"""
    task_metrics["duration_sec"] = round(
        time.time() - task_metrics["started_at"], 2
    )
    export_task_trace(task_metrics)


async def execute_task(task_id: str):
    """
    Enhanced orchestration loop with Week 2 memory + your custom metrics
    
    YOUR ENHANCEMENTS:
    - Detailed metrics tracking
    - Failure classification
    - Step trace export
    - Duration tracking
    
    WEEK 2 FEATURES:
    - Memory agent recalls similar past tasks
    - File persistence across tasks
    - Enhanced context sharing
    - Post-execution learning
    """
    
    # YOUR METRICS TRACKING
    task_metrics = {
        "task_id": task_id,
        "started_at": time.time(),
        "total_steps": 0,
        "completed_steps": 0,
        "retries": 0,
        "failures": [],
        "step_traces": [],
        "memories_used": [],  # Week 2
        "created_files": [],  # Week 2
    }
    
    logger.info("orchestrator_started", task_id=task_id)
    
    # Initialize agents
    planner = PlannerAgent()
    executor = ExecutorAgent()
    critic = CriticAgent()
    
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
            
            # WEEK 2: Initialize memory agent
            memory_agent = MemoryAgent(db=db)
            
            # WEEK 2: Create task context
            task_context = TaskContext(
                id=str(uuid.uuid4()),
                task_id=task_id,
                context_data={},
                created_files=[],
                memories_used=[]
            )
            db.add(task_context)
            db.commit()
            
            logger.info("orchestrator_planning", task=task.user_input)
            
            # WEEK 2: Recall similar past tasks
            similar_memories = await memory_agent.recall_similar_tasks(
                task_description=task.user_input,
                limit=3
            )
            
            if similar_memories:
                task_context.memories_used = [m["id"] for m in similar_memories]
                task_metrics["memories_used"] = task_context.memories_used
                db.commit()
                
                logger.info(
                    "orchestrator_memories_recalled",
                    num_memories=len(similar_memories)
                )
                
                # Get suggested approach
                suggestion = await memory_agent.suggest_approach(
                    task_description=task.user_input,
                    similar_memories=similar_memories
                )
                
                if suggestion:
                    logger.info(
                        "orchestrator_suggestion",
                        suggestion=suggestion
                    )
            
            # PHASE 1: PLANNING
            try:
                plan = await planner.plan(task.user_input)
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
            
            db.commit()
            
            logger.info("orchestrator_plan_created", num_steps=len(plan))
            task_metrics["total_steps"] = len(plan)
            
            # PHASE 2: EXECUTION with enhanced context
            context: Dict[str, Any] = {
                "task_description": task.user_input,
                "memories": similar_memories  # Week 2
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
                    
                    # Execute step with full context
                    try:
                        tool_result = await executor.execute_step(
                            instruction=step.instruction,
                            context=context
                        )
                        
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
                        evaluation = await critic.evaluate(
                            step_instruction=step.instruction,
                            tool_result=tool_result,
                            retry_count=retry_count
                        )
                        
                        logger.info(
                            "orchestrator_step_evaluated",
                            step_number=step_number,
                            verdict=evaluation.verdict,
                            reason=evaluation.reason
                        )
                        
                        # YOUR METRICS: Track step trace
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
                            
                            # Add output to context for next steps
                            context[f"step_{step_number}_output"] = tool_result.output
                            context[f"step_{step_number}_success"] = True
                            
                            # WEEK 2: Track created files
                            if tool_result.metadata.get("filename"):
                                filename = tool_result.metadata["filename"]
                                if filename not in task_context.created_files:
                                    task_context.created_files.append(filename)
                                    task_metrics["created_files"].append(filename)
                        
                        elif evaluation.verdict == Verdict.RETRY:
                            task_metrics["retries"] += 1
                            step.status = StepStatus.RETRYING
                            retry_count += 1
                            
                            logger.warning(
                                "orchestrator_step_retrying",
                                step_number=step_number,
                                retry_count=retry_count,
                                suggestions=evaluation.suggestions
                            )
                            
                            # Wait before retry
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
                            
                            # YOUR METRICS: Track failure
                            task_metrics["failures"].append({
                                "step_number": step_number,
                                "error": step.error,
                                "category": classify_failure(step.error),
                            })
                            
                            db.commit()
                            
                            # WEEK 2: Learn from failure
                            await memory_agent.store_task_memory(task)
                            
                            finalize_and_export(task_metrics)
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
                        
                        # Fail task on exception
                        task.status = TaskStatus.FAILED
                        task.error_message = f"Step {step_number} crashed: {str(e)}"
                        task.completed_at = datetime.utcnow()
                        
                        # YOUR METRICS: Track crash
                        task_metrics["failures"].append({
                            "step_number": step_number,
                            "error": str(e),
                            "category": "ORCHESTRATOR_ERROR",
                        })
                        
                        db.commit()
                        
                        # WEEK 2: Learn from crash
                        await memory_agent.store_task_memory(task)
                        
                        finalize_and_export(task_metrics)
                        return
                
                # If step didn't succeed after retries
                if not step_succeeded:
                    logger.error(
                        "orchestrator_step_exhausted_retries",
                        step_number=step_number
                    )
                    
                    step.status = StepStatus.FAILED
                    task.status = TaskStatus.FAILED
                    task.error_message = f"Step {step_number} exhausted retries"
                    task.completed_at = datetime.utcnow()
                    
                    # YOUR METRICS: Track retry exhaustion
                    task_metrics["failures"].append({
                        "step_number": step_number,
                        "error": "Exhausted retries",
                        "category": "RETRY_LIMIT_EXCEEDED",
                    })
                    
                    db.commit()
                    
                    # WEEK 2: Learn from retry exhaustion
                    await memory_agent.store_task_memory(task)
                    
                    finalize_and_export(task_metrics)
                    return
            
            # PHASE 3: COMPLETION
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            # WEEK 2: Update context
            task_context.context_data = context
            db.commit()
            
            # WEEK 2: Learn from success
            memory_id = await memory_agent.store_task_memory(task)
            if memory_id:
                logger.info("orchestrator_learned", memory_id=memory_id)
            
            finalize_and_export(task_metrics)
            logger.info("orchestrator_completed", task_id=task_id)
    
    except Exception as e:
        logger.error("orchestrator_error", task_id=task_id, error=str(e))
        
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