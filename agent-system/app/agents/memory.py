import json
import uuid
import structlog
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.utils.llm import call_llm
from app.models.memory import Memory
from app.models.task import Task, Step

logger = structlog.get_logger()

class MemoryAgent:
    """
    Learns from past executions and provides context-aware suggestions.
    
    CAPABILITIES:
    1. Store successful patterns
    2. Recall similar past tasks
    3. Suggest proven approaches
    4. Warn about known failure modes
    
    USES: Claude Haiku (same as executor for consistency)
    """
    
    SYSTEM_PROMPT = """You are a memory and learning agent. Your job is to:
1. Identify patterns in task execution
2. Extract reusable strategies
3. Classify task types

When analyzing a completed task, extract:
- General pattern (e.g., "file_creation", "calculation", "data_processing")
- Key strategy that led to success
- Tools that were effective
- Common pitfalls to avoid

RESPONSE FORMAT (JSON only):
{
  "pattern_type": "success|failure",
  "task_pattern": "general_category",
  "strategy": "what worked or failed",
  "tools_used": ["tool1", "tool2"],
  "key_insights": "what to remember"
}

RESPOND ONLY WITH JSON."""

    def __init__(self, db: Session, model: str = "claude-haiku-4-5-20251001"):
        self.db = db
        self.model = model  # Now using Claude by default
    
    async def store_task_memory(self, task: Task) -> Optional[str]:
        """
        Learn from a completed task
        
        Args:
            task: Completed task to learn from
            
        Returns:
            Memory ID if stored, None if not worth storing
        """
        if task.status not in ["COMPLETED", "FAILED"]:
            return None
        
        logger.info("memory_storing", task_id=task.id, status=task.status)
        
        # Prepare task summary
        steps_summary = []
        for step in sorted(task.steps, key=lambda s: s.step_number):
            steps_summary.append({
                "instruction": step.instruction,
                "tool": step.tool_name,
                "status": step.status,
                "retry_count": step.retry_count
            })
        
        task_summary = {
            "user_input": task.user_input,
            "status": task.status,
            "steps": steps_summary,
            "error": task.error_message
        }
        
        # Ask LLM to extract patterns
        user_prompt = f"""Analyze this completed task and extract learnings:

{json.dumps(task_summary, indent=2)}

Extract reusable patterns and strategies. Return JSON only."""
        
        try:
            response = await call_llm(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.2
            )
            
            # Parse response (Claude formatting)
            response_text = response.strip()
            
            # Handle markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1] if "\n" in response_text else response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
            
            # Extract JSON object
            start = response_text.find("{")
            end = response_text.rfind("}")
            
            if start != -1 and end != -1 and end > start:
                response_text = response_text[start:end+1]
            
            analysis = json.loads(response_text)
            
            # Create memory record
            memory = Memory(
                id=str(uuid.uuid4()),
                pattern_type=analysis.get("pattern_type", "success"),
                task_pattern=analysis.get("task_pattern", "general"),
                original_task_id=task.id,
                task_description=task.user_input,
                strategy=analysis.get("strategy"),
                tools_used=analysis.get("tools_used", []),
                steps_taken=steps_summary,
                error_message=task.error_message if task.status == "FAILED" else None,
                failure_reason=analysis.get("strategy") if task.status == "FAILED" else None,
                success_rate=1.0 if task.status == "COMPLETED" else 0.0,
                avg_steps=float(len(task.steps)) if task.steps else 0.0
            )
            
            self.db.add(memory)
            self.db.commit()
            
            logger.info(
                "memory_stored",
                memory_id=memory.id,
                pattern=memory.task_pattern
            )
            
            return memory.id
        
        except json.JSONDecodeError as e:
            logger.error("memory_store_json_error", error=str(e), response=response_text[:500])
            return None
        
        except Exception as e:
            logger.error("memory_store_failed", error=str(e))
            return None
    
    async def recall_similar_tasks(
        self,
        task_description: str,
        limit: int = 3
    ) -> List[Dict]:
        """
        Find similar past tasks and their strategies
        
        Args:
            task_description: Current task to find matches for
            limit: Max number of memories to return
            
        Returns:
            List of relevant memories with strategies
        """
        logger.info("memory_recalling", task=task_description)
        
        # Get successful memories from database
        memories = self.db.query(Memory).filter(
            Memory.pattern_type == "success"
        ).order_by(
            Memory.success_rate.desc(),
            Memory.times_referenced.desc()
        ).limit(limit * 2).all()  # Get more than needed for filtering
        
        if not memories:
            logger.info("memory_recall_empty")
            return []
        
        # Use Claude to find most relevant
        memories_summary = []
        for mem in memories:
            memories_summary.append({
                "id": mem.id,
                "task_pattern": mem.task_pattern,
                "task": mem.task_description,
                "strategy": mem.strategy,
                "tools": mem.tools_used
            })
        
        prompt = f"""Current task: {task_description}

Past successful tasks:
{json.dumps(memories_summary, indent=2)}

Select the {limit} most relevant past tasks that could help with the current one.
Return JSON array of memory IDs in order of relevance:
{{"relevant_ids": ["id1", "id2", ...]}}

Return ONLY the JSON object, no other text."""
        
        try:
            response = await call_llm(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.1
            )
            
            # Parse response (Claude formatting)
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
            
            result = json.loads(response_text)
            relevant_ids = result.get("relevant_ids", [])[:limit]
            
            # Get full memory objects and update usage
            relevant_memories = []
            for mem_id in relevant_ids:
                mem = self.db.query(Memory).filter(Memory.id == mem_id).first()
                if mem:
                    mem.times_referenced += 1
                    mem.last_used = datetime.utcnow()
                    
                    relevant_memories.append({
                        "id": mem.id,
                        "pattern": mem.task_pattern,
                        "strategy": mem.strategy,
                        "tools": mem.tools_used,
                        "success_rate": mem.success_rate
                    })
            
            self.db.commit()
            
            logger.info(
                "memory_recalled",
                num_memories=len(relevant_memories)
            )
            
            return relevant_memories
        
        except json.JSONDecodeError as e:
            logger.error("memory_recall_json_error", error=str(e), response=response_text[:500])
            return []
        
        except Exception as e:
            logger.error("memory_recall_failed", error=str(e))
            return []
    
    async def suggest_approach(
        self,
        task_description: str,
        similar_memories: List[Dict]
    ) -> Optional[str]:
        """
        Generate approach suggestion based on memories
        
        Args:
            task_description: Current task
            similar_memories: Relevant past tasks
            
        Returns:
            Suggested approach or None
        """
        if not similar_memories:
            return None
        
        prompt = f"""Task: {task_description}

Successful past approaches:
{json.dumps(similar_memories, indent=2)}

Based on these past successes, suggest a high-level approach for the current task.
Be specific about tools and strategies. Keep it under 3 sentences."""
        
        try:
            response = await call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3
            )
            
            logger.info("memory_suggestion_generated")
            return response.strip()
        
        except Exception as e:
            logger.error("memory_suggestion_failed", error=str(e))
            return None