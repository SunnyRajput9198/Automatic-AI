import json
import uuid
import structlog
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.utils.llm import call_llm
from app.models.memory import Memory
from app.agents.reflection import Reflection

logger = structlog.get_logger()


class ConfidenceMemory:
    """
    Enhanced memory system with confidence tracking.
    
    KEY FEATURES:
    1. Each memory has confidence score (0-1)
    2. Confidence updates from reflection
    3. Retrieval weighted by: relevance × confidence × recency
    4. Success/failure tracking per pattern
    """
    
    def __init__(self, db: Session, model: str = "claude-haiku-4-5-20251001"):
        self.db = db
        self.model = model
    
    async def store_with_confidence(
        self,
        pattern_type: str,
        task_pattern: str,
        task_id: str,
        task_description: str,
        strategy: str,
        tools_used: List[str],
        steps_taken: List[Dict],
        success: bool,
        initial_confidence: float = 0.5,
        reflection: Optional[Reflection] = None
    ) -> str:
        """
        Store memory with confidence tracking.
        
        Args:
            pattern_type: "success" or "failure"
            task_pattern: Pattern category
            task_id: Original task ID
            task_description: Task description
            strategy: What strategy was used
            tools_used: List of tools
            steps_taken: Step details
            success: Whether task succeeded
            initial_confidence: Starting confidence (0-1)
            reflection: Optional reflection to incorporate
            
        Returns:
            Memory ID
        """
        logger.info(
            "confidence_memory_storing",
            pattern=task_pattern,
            success=success,
            confidence=initial_confidence
        )
        
        # Adjust confidence based on reflection if available
        final_confidence = initial_confidence
        if reflection:
            # If pattern quality is provided, use it
            if reflection.pattern_quality > 0:
                final_confidence = reflection.pattern_quality
        
        # Create memory with confidence
        memory = Memory(
            id=str(uuid.uuid4()),
            pattern_type=pattern_type,
            task_pattern=task_pattern,
            original_task_id=task_id,
            task_description=task_description,
            strategy=strategy,
            tools_used=tools_used,
            steps_taken=steps_taken,
            error_message=None if success else "Task failed",
            failure_reason=None if success else strategy,
            success_rate=final_confidence if success else (1 - final_confidence),
            avg_steps=float(len(steps_taken)),
            times_referenced=0,
            last_used=None
        )
        
        self.db.add(memory)
        self.db.commit()
        
        logger.info(
            "confidence_memory_stored",
            memory_id=memory.id,
            final_confidence=final_confidence
        )
        
        return memory.id
    
    async def update_confidence_from_reflection(
        self,
        reflection: Reflection,
        task_pattern: str
    ):
        """
        Update confidence scores based on reflection.
        
        Args:
            reflection: Reflection output with confidence updates
            task_pattern: The pattern this task represents
        """
        if not reflection.confidence_updates:
            return
        
        logger.info(
            "confidence_updating",
            num_updates=len(reflection.confidence_updates)
        )
        
        # Update memories matching the patterns
        for pattern, confidence_change in reflection.confidence_updates.items():
            # Find recent memories with this pattern
            memories = self.db.query(Memory).filter(
                Memory.task_pattern.like(f"%{pattern}%")
            ).order_by(
                desc(Memory.created_at)
            ).limit(5).all()
            
            for memory in memories:
                # Update success rate (our proxy for confidence)
                old_confidence = memory.success_rate
                new_confidence = max(0.0, min(1.0, old_confidence + confidence_change))
                
                memory.success_rate = new_confidence
                
                logger.debug(
                    "confidence_updated",
                    memory_id=memory.id,
                    pattern=pattern,
                    old=old_confidence,
                    new=new_confidence,
                    change=confidence_change
                )
            
            self.db.commit()
    
    def calculate_recency_score(self, memory: Memory) -> float:
        """
        Calculate recency score (0-1).
        More recent = higher score.
        """
        if not memory.last_used and not memory.created_at:
            return 0.5
        
        last_time = memory.last_used or memory.created_at
        now = datetime.utcnow()
        days_old = (now - last_time).days
        
        # Exponential decay: 1.0 at 0 days, 0.5 at 30 days, 0.1 at 90 days
        recency = max(0.1, 1.0 / (1.0 + days_old / 30.0))
        
        return recency
    
    async def recall_with_confidence(
        self,
        task_description: str,
        min_confidence: float = 0.3,
        limit: int = 3
    ) -> tuple[List[Dict], float]:
        """
        Retrieve memories weighted by relevance × confidence × recency.
        
        Args:
            task_description: Current task
            min_confidence: Minimum confidence threshold
            limit: Max memories to return
            
        Returns:
            (memories, avg_confidence)
        """
        logger.info("confidence_recall_starting", task=task_description)
        
        # Get all successful memories above confidence threshold
        candidate_memories = self.db.query(Memory).filter(
            Memory.pattern_type == "success",
            Memory.success_rate >= min_confidence
        ).order_by(
            desc(Memory.success_rate),  # Confidence first
            desc(Memory.times_referenced),  # Then usage
            desc(Memory.created_at)  # Then recency
        ).limit(limit * 3).all()  # Get extras for LLM filtering
        
        if not candidate_memories:
            logger.info("confidence_recall_empty")
            return [], 0.0
        
        # Calculate composite scores
        scored_memories = []
        for mem in candidate_memories:
            confidence = mem.success_rate
            recency = self.calculate_recency_score(mem)
            
            scored_memories.append({
                "id": mem.id,
                "pattern": mem.task_pattern,
                "task": mem.task_description,
                "strategy": mem.strategy,
                "tools": mem.tools_used,
                "confidence": confidence,
                "recency": recency,
                "times_used": mem.times_referenced,
                # Composite score: confidence × recency × sqrt(usage)
                "composite_score": confidence * recency * (1 + (mem.times_referenced ** 0.5) / 10)
            })
        
        # Sort by composite score
        scored_memories.sort(key=lambda x: x["composite_score"], reverse=True)
        
        # Use LLM to find most relevant
        top_candidates = scored_memories[:limit * 2]
        
        prompt = f"""Current task: {task_description}

Past successful patterns (with confidence scores):
{json.dumps(top_candidates, indent=2)}

Select the {limit} most relevant patterns for this task.
Consider both similarity and confidence.
Return JSON array of memory IDs:
{{"relevant_ids": ["id1", "id2", ...]}}

ONLY return JSON."""
        
        try:
            response = await call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.strip()
            
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1] if "\n" in response_text else response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
            
            start = response_text.find("{")
            end = response_text.rfind("}")
            
            if start != -1 and end != -1:
                response_text = response_text[start:end+1]
            
            result = json.loads(response_text)
            relevant_ids = result.get("relevant_ids", [])[:limit]
            
            # Get full memory data and update usage
            relevant_memories = []
            confidences = []
            
            for mem_id in relevant_ids:
                # Find in our scored list
                mem_data = next((m for m in scored_memories if m["id"] == mem_id), None)
                if mem_data:
                    # Update DB
                    mem_obj = self.db.query(Memory).filter(Memory.id == mem_id).first()
                    if mem_obj:
                        mem_obj.times_referenced += 1
                        mem_obj.last_used = datetime.utcnow()
                    
                    relevant_memories.append(mem_data)
                    confidences.append(mem_data["confidence"])
            
            self.db.commit()
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            logger.info(
                "confidence_recall_completed",
                num_memories=len(relevant_memories),
                avg_confidence=avg_confidence
            )
            
            return relevant_memories, avg_confidence
        
        except Exception as e:
            logger.error("confidence_recall_error", error=str(e))
            # Fallback: return top by composite score
            fallback_memories = scored_memories[:limit]
            avg_conf = sum(m["confidence"] for m in fallback_memories) / len(fallback_memories) if fallback_memories else 0.0
            return fallback_memories, avg_conf