import time
import json
import structlog
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logger = structlog.get_logger()


@dataclass
class LLMCall:
    """Single LLM call record"""
    agent: str  # Which agent made the call
    model: str
    timestamp: float
    duration_ms: float
    tokens_estimated: int  # Rough estimate based on response length
    purpose: str  # "reasoning", "planning", "execution", etc.
    
    def __post_init__(self):
        self.timestamp=datetime.now().timestamp()

@dataclass
class TaskCost:
    """Complete cost breakdown for a task"""
    task_id: str
    started_at: float
    completed_at: Optional[float]
    
    # LLM costs
    llm_calls: List[LLMCall]
    total_llm_calls: int
    reasoning_calls: int
    planning_calls: int
    execution_calls: int
    critic_calls: int
    reflection_calls: int
    
    # Execution costs
    total_retries: int
    total_steps: int
    search_operations: int
    
    # Time costs
    duration_sec: float
    
    # Outcome
    success: bool
    
    # Efficiency metrics
    cost_per_step: float
    llm_efficiency: float  # steps / llm_calls (higher is better)


class CostTracker:
    """
    Track and analyze costs for agent operations.
    
    GOALS:
    1. Understand where time/money is spent
    2. Identify inefficiencies
    3. Learn cost-effective patterns
    4. Enable cost-aware decision making
    """
    
    # Rough token estimates (very approximate)
    TOKENS_PER_CHAR = 0.25  # ~4 chars per token
    COST_PER_1M_TOKENS = {
        "claude-haiku-4-5-20251001": 0.25,  # $0.25 per 1M input tokens
        "claude-sonnet-4-5-20250929": 3.00,  # $3.00 per 1M input tokens
    }
    
    def __init__(self):
        self.current_task: Optional[TaskCost] = None
        self.completed_tasks: List[TaskCost] = []
        
        # Create cost tracking directory
        Path("costs").mkdir(exist_ok=True)
    
    def start_task(self, task_id: str):
        """Begin tracking a new task"""
        self.current_task = TaskCost(
            task_id=task_id,
            started_at=time.time(),
            completed_at=None,
            llm_calls=[],
            total_llm_calls=0,
            reasoning_calls=0,
            planning_calls=0,
            execution_calls=0,
            critic_calls=0,
            reflection_calls=0,
            total_retries=0,
            total_steps=0,
            search_operations=0,
            duration_sec=0.0,
            success=False,
            cost_per_step=0.0,
            llm_efficiency=0.0
        )
        
        logger.info("cost_tracking_started", task_id=task_id)
    
    def record_llm_call(
        self,
        agent: str,
        model: str,
        response_length: int,
        purpose: str,
        duration_ms: float
    ):
        """Record an LLM call"""
        if not self.current_task:
            logger.warning("cost_tracker_no_active_task")
            return
        
        tokens_est = int(response_length * self.TOKENS_PER_CHAR)
        
        call = LLMCall(
            agent=agent,
            model=model,
            timestamp=time.time(),
            duration_ms=duration_ms,
            tokens_estimated=tokens_est,
            purpose=purpose
        )
        
        self.current_task.llm_calls.append(call)
        self.current_task.total_llm_calls += 1
        
        # Categorize
        if purpose == "reasoning":
            self.current_task.reasoning_calls += 1
        elif purpose == "planning":
            self.current_task.planning_calls += 1
        elif purpose == "execution":
            self.current_task.execution_calls += 1
        elif purpose == "critic":
            self.current_task.critic_calls += 1
        elif purpose == "reflection":
            self.current_task.reflection_calls += 1
        
        logger.debug(
            "llm_call_recorded",
            agent=agent,
            purpose=purpose,
            tokens=tokens_est
        )
    
    def record_retry(self):
        """Record a retry"""
        if self.current_task:
            self.current_task.total_retries += 1
    
    def record_step(self):
        """Record a step execution"""
        if self.current_task:
            self.current_task.total_steps += 1
    
    def record_search(self):
        """Record a web search operation"""
        if self.current_task:
            self.current_task.search_operations += 1
    
    def complete_task(self, success: bool):
        """Finalize task tracking"""
        if not self.current_task:
            logger.warning("cost_tracker_no_active_task")
            return
        
        self.current_task.completed_at = time.time()
        self.current_task.duration_sec = round(
            self.current_task.completed_at - self.current_task.started_at,
            2
        )
        self.current_task.success = success
        
        # Calculate efficiency metrics
        if self.current_task.total_steps > 0:
            self.current_task.cost_per_step = round(
                self.current_task.total_llm_calls / self.current_task.total_steps,
                2
            )
            
            self.current_task.llm_efficiency = round(
                self.current_task.total_steps / max(1, self.current_task.total_llm_calls),
                2
            )
        
        # Save to completed list
        self.completed_tasks.append(self.current_task)
        
        # Export
        self._export_task_cost(self.current_task)
        
        logger.info(
            "cost_tracking_completed",
            task_id=self.current_task.task_id,
            duration=self.current_task.duration_sec,
            llm_calls=self.current_task.total_llm_calls,
            efficiency=self.current_task.llm_efficiency
        )
        
        self.current_task = None
    
    def _export_task_cost(self, task_cost: TaskCost):
        """Export task cost to JSON file"""
        filename = f"costs/task_{task_cost.task_id}.json"
        
        # Convert to dict
        data = asdict(task_cost)
        
        # Add estimated monetary cost
        total_tokens = sum(call.tokens_estimated for call in task_cost.llm_calls)
        
        # Estimate cost by model
        cost_by_model = {}
        for call in task_cost.llm_calls:
            model = call.model
            if model not in cost_by_model:
                cost_by_model[model] = 0
            cost_by_model[model] += call.tokens_estimated
        
        total_cost_usd = 0.0
        for model, tokens in cost_by_model.items():
            cost_per_million = self.COST_PER_1M_TOKENS.get(model, 1.0)
            cost_usd = (tokens / 1_000_000) * cost_per_million
            total_cost_usd += cost_usd
        
        data["total_tokens_estimated"] = total_tokens
        data["estimated_cost_usd"] = round(total_cost_usd, 6)
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.debug("cost_export_saved", filename=filename)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked tasks"""
        if not self.completed_tasks:
            return {
                "total_tasks": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
                "avg_llm_calls": 0.0,
                "avg_efficiency": 0.0
            }
        
        total = len(self.completed_tasks)
        successes = sum(1 for t in self.completed_tasks if t.success)
        
        return {
            "total_tasks": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": round(successes / total, 2),
            "avg_duration": round(
                sum(t.duration_sec for t in self.completed_tasks) / total,
                2
            ),
            "avg_llm_calls": round(
                sum(t.total_llm_calls for t in self.completed_tasks) / total,
                2
            ),
            "avg_retries": round(
                sum(t.total_retries for t in self.completed_tasks) / total,
                2
            ),
            "avg_efficiency": round(
                sum(t.llm_efficiency for t in self.completed_tasks) / total,
                2
            ),
            "total_searches": sum(t.search_operations for t in self.completed_tasks)
        }
    
    def compare_tasks(self, task_id_1: str, task_id_2: str) -> Dict[str, Any]:
        """Compare two tasks for efficiency analysis"""
        task1 = next((t for t in self.completed_tasks if t.task_id == task_id_1), None)
        task2 = next((t for t in self.completed_tasks if t.task_id == task_id_2), None)
        
        if not task1 or not task2:
            return {"error": "One or both tasks not found"}
        
        return {
            "task_1": task_id_1,
            "task_2": task_id_2,
            "duration_improvement": round(
                ((task1.duration_sec - task2.duration_sec) / task1.duration_sec) * 100,
                1
            ) if task1.duration_sec > 0 else 0,
            "llm_call_reduction": task1.total_llm_calls - task2.total_llm_calls,
            "retry_reduction": task1.total_retries - task2.total_retries,
            "efficiency_gain": round(task2.llm_efficiency - task1.llm_efficiency, 2)
        }


# Global cost tracker instance
global_cost_tracker = CostTracker()