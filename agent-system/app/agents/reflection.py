import json
import uuid
import structlog
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy.orm import Session

from app.utils.llm import call_llm
from app.models.task import Task

logger = structlog.get_logger()


class Reflection(BaseModel):
    """Structured reflection output"""
    what_worked: List[str]  # Successful strategies
    what_failed: List[str]  # Failures and mistakes
    root_causes: List[str]  # Why failures happened
    lessons: List[str]  # Key takeaways
    confidence_updates: Dict[str, float]  # Pattern -> confidence change
    improvement_suggestions: List[str]  # How to do better next time
    pattern_quality: float  # How reusable is this pattern (0-1)


class ReflectionAgent:
    """
    Post-task reflection and learning agent.
    
    RUNS AFTER: Task completion (success or failure)
    PURPOSE: Extract lessons and improve future performance
    MODEL: Claude Haiku (analysis + learning)
    """
    
    SYSTEM_PROMPT = """You are a reflection and learning agent. Your job is to analyze completed tasks and extract actionable lessons.

Your analysis helps the system:
1. Learn from successes (what to repeat)
2. Learn from failures (what to avoid)
3. Improve planning quality over time
4. Build reusable patterns
5. Increase confidence in proven approaches

REFLECTION FRAMEWORK:

What to Analyze:
- Did the plan work as expected?
- Were the right tools chosen?
- Did retries help or waste time?
- Were there unnecessary steps?
- What surprised us?

Success Patterns:
- Tool combinations that worked well
- Planning strategies that led to success
- Efficient approaches (few retries, fast completion)
- Reusable patterns for similar tasks

Failure Analysis:
- What actually broke?
- Was it planning, execution, or tool selection?
- Could we have predicted this?
- What would we do differently?

Confidence Updates:
- Increase confidence for patterns that worked (+0.1 to +0.3)
- Decrease confidence for patterns that failed (-0.1 to -0.3)
- Scale changes by how certain we are about the cause

Pattern Quality Assessment:
- 0.9-1.0: Highly reusable, clear pattern, works consistently
- 0.7-0.9: Good pattern, some edge cases
- 0.5-0.7: Situational, works in specific contexts
- 0.3-0.5: Experimental, needs more data
- 0.0-0.3: Unreliable, too specific, or failed

RESPONSE FORMAT (JSON only):
{
  "what_worked": ["strategy1", "strategy2"],
  "what_failed": ["failure1", "failure2"],
  "root_causes": ["cause1", "cause2"],
  "lessons": ["lesson1", "lesson2"],
  "confidence_updates": {
    "pattern_name": 0.2,
    "another_pattern": -0.1
  },
  "improvement_suggestions": ["suggestion1", "suggestion2"],
  "pattern_quality": 0.85
}

EXAMPLES:

Successful file creation task:
{
  "what_worked": [
    "file_write tool worked on first try",
    "Simple one-step plan was efficient"
  ],
  "what_failed": [],
  "root_causes": [],
  "lessons": [
    "file_write is reliable for basic file creation",
    "No need to check file existence for new files"
  ],
  "confidence_updates": {
    "file_write_basic": 0.1
  },
  "improvement_suggestions": [
    "Could skip existence check for 'create' tasks"
  ],
  "pattern_quality": 0.95
}

Failed calculation with retries:
{
  "what_worked": [
    "python_executor was correct tool choice",
    "Retry logic caught the first error"
  ],
  "what_failed": [
    "Initial code had recursion depth error",
    "Second attempt still used recursion"
  ],
  "root_causes": [
    "Planner didn't anticipate recursion limits for large numbers",
    "Executor didn't learn from first retry"
  ],
  "lessons": [
    "For large fibonacci, must use iterative approach",
    "Retries need better error analysis, not just regeneration"
  ],
  "confidence_updates": {
    "python_recursion_large_numbers": -0.2,
    "python_iterative_fibonacci": 0.3
  },
  "improvement_suggestions": [
    "Add recursion limit checks in planning",
    "Critic should suggest iterative approach after recursion error"
  ],
  "pattern_quality": 0.6
}

RESPOND ONLY WITH JSON."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
    
    async def reflect(
        self,
        task: Task,
        reasoning_used: Optional[Dict] = None,
        search_used: bool = False
    ) -> Reflection:
        """
        Analyze completed task and generate reflection.
        
        Args:
            task: Completed task (success or failure)
            reasoning_used: Optional reasoning output that was used
            search_used: Whether web search was used
            
        Returns:
            Reflection with lessons and confidence updates
        """
        logger.info(
            "reflection_starting",
            task_id=task.id,
            status=task.status
        )
        
        # Build task summary
        steps_summary = []
        total_retries = 0
        
        for step in sorted(task.steps, key=lambda s: s.step_number):
            total_retries += step.retry_count
            steps_summary.append({
                "step": step.step_number,
                "instruction": step.instruction,
                "tool": step.tool_name,
                "status": step.status,
                "retries": step.retry_count,
                "error": step.error if step.error else None
            })
        
        # Calculate efficiency metrics
        duration_sec = None
        if task.completed_at and task.created_at:
            duration_sec = (task.completed_at - task.created_at).total_seconds()
        
        task_analysis = {
            "task": task.user_input,
            "status": task.status,
            "total_steps": len(task.steps),
            "total_retries": total_retries,
            "duration_sec": duration_sec,
            "steps": steps_summary,
            "final_error": task.error_message,
            "reasoning_used": reasoning_used,
            "search_used": search_used
        }
        
        user_prompt = f"""COMPLETED TASK ANALYSIS:

{json.dumps(task_analysis, indent=2, default=str)}

Reflect on this task execution:
1. What strategies worked well?
2. What failed and why?
3. What should we remember for similar future tasks?
4. How should we adjust our confidence in different approaches?
5. How reusable is the pattern from this task?

Be specific and actionable. Return JSON only."""
        
        try:
            response = await call_llm(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.3,  # Some creativity in analysis
            )
            
            # Parse response
            response_text = response.strip()
            
            # Handle markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1] if "\n" in response_text else response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
            
            # Extract JSON
            start = response_text.find("{")
            end = response_text.rfind("}")
            
            if start != -1 and end != -1 and end > start:
                response_text = response_text[start:end+1]
            
            reflection_data = json.loads(response_text)
            
            # Create reflection object
            reflection = Reflection(
                what_worked=reflection_data.get("what_worked", []),
                what_failed=reflection_data.get("what_failed", []),
                root_causes=reflection_data.get("root_causes", []),
                lessons=reflection_data.get("lessons", []),
                confidence_updates=reflection_data.get("confidence_updates", {}),
                improvement_suggestions=reflection_data.get("improvement_suggestions", []),
                pattern_quality=float(reflection_data.get("pattern_quality", 0.5))
            )
            
            logger.info(
                "reflection_completed",
                num_lessons=len(reflection.lessons),
                pattern_quality=reflection.pattern_quality,
                confidence_changes=len(reflection.confidence_updates)
            )
            
            return reflection
        
        except json.JSONDecodeError as e:
            logger.error("reflection_json_error", error=str(e), response=response_text[:500])
            # Return empty reflection
            return Reflection(
                what_worked=[],
                what_failed=["Reflection analysis failed"],
                root_causes=["JSON parsing error"],
                lessons=[],
                confidence_updates={},
                improvement_suggestions=[],
                pattern_quality=0.0
            )
        
        except Exception as e:
            logger.error("reflection_error", error=str(e))
            # Return empty reflection
            return Reflection(
                what_worked=[],
                what_failed=["Reflection error"],
                root_causes=[str(e)],
                lessons=[],
                confidence_updates={},
                improvement_suggestions=[],
                pattern_quality=0.0
            )