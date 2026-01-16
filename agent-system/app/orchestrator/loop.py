import asyncio
import uuid
import structlog
from datetime import datetime
from typing import Dict, Any

from app.db.session import get_db_context
from app.models.task import Task, Step, TaskStatus, StepStatus
from app.agents.planner import PlannerAgent
from app.agents.executor import ExecutorAgent
from app.agents.critic import CriticAgent, Verdict

logger = structlog.get_logger()

async def execute_task(task_id: str):
    """
    Main orchestration loop - THE HEART OF THE SYSTEM
    
    Flow:
    1. Planner breaks down task
    2. For each step:
       a. Executor runs it
       b. Critic evaluates
       c. Retry if needed
    3. Mark task complete/failed
    
    This is what makes the system autonomous.
    """
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
            
            logger.info("orchestrator_planning", task=task.user_input)
            
            # PHASE 1: PLANNING
            try:
                plan = await planner.plan(task.user_input)
            except Exception as e:
                logger.error("orchestrator_planning_failed", error=str(e))
                task.status = TaskStatus.FAILED
                task.error_message = f"Planning failed: {str(e)}"
                db.commit()
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
            
            # PHASE 2: EXECUTION
            context: Dict[str, Any] = {}
            
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
                
                while retry_count <= max_retries and not step_succeeded:
                    step.status = StepStatus.RUNNING
                    step.retry_count = retry_count
                    db.commit()
                    
                    # Execute step
                    try:
                        tool_result = await executor.execute_step(
                            instruction=step.instruction,
                            context=context
                        )
                        
                        # Store result
                        step.result = tool_result.output
                        step.error = tool_result.error
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
                        
                        # Handle verdict
                        if evaluation.verdict == Verdict.PASS:
                            step.status = StepStatus.COMPLETED
                            step.completed_at = datetime.utcnow()
                            step_succeeded = True
                            
                            # Add output to context for next steps
                            context[f"step_{step_number}_output"] = tool_result.output
                        
                        elif evaluation.verdict == Verdict.RETRY:
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
                            db.commit()
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
                        db.commit()
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
                    db.commit()
                    return
            
            # PHASE 3: COMPLETION
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            db.commit()
            
            logger.info("orchestrator_completed", task_id=task_id)
    
    except Exception as e:
        logger.error("orchestrator_error", task_id=task_id, error=str(e))
        
        with get_db_context() as db:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = TaskStatus.FAILED
                task.error_message = f"Orchestrator error: {str(e)}"
                task.completed_at = datetime.utcnow()
                db.commit()