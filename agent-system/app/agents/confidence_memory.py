import json
import uuid
import structlog
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.utils.llm import call_llm
from app.models.memory import Memory
from app.agents.reflection import Reflection

logger = structlog.get_logger()


class ConfidenceMemory:
    """
    Memory system with per-pattern confidence tracking.

    Each memory stored in PostgreSQL carries a `success_rate` field used
    as a confidence proxy (0–1).  Retrieval is weighted by:

        composite = confidence × recency × (1 + √usage / 10)

    The LLM then picks the most task-relevant subset from the top
    composite-scored candidates.
    """

    def __init__(self, db: Session, model: str = "claude-haiku-4-5-20251001"):
        self.db = db
        self.model = model

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

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
        reflection: Optional[Reflection] = None,
    ) -> str:
        """
        Persist a task outcome as a memory record.

        The stored `success_rate` is:
        - For successes: reflection.pattern_quality if available, else initial_confidence
        - For failures:  0.0  (failed patterns carry no positive confidence)
        """
        logger.info(
            "confidence_memory_storing",
            pattern=task_pattern,
            success=success,
            initial_confidence=initial_confidence,
        )

        if not success:
            # Failed patterns are stored with zero confidence so they are
            # never surfaced by the min_confidence filter in recall.
            final_confidence = 0.0
        elif reflection and reflection.pattern_quality > 0:
            final_confidence = reflection.pattern_quality
        else:
            final_confidence = initial_confidence

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
            success_rate=final_confidence,
            avg_steps=float(len(steps_taken)),
            times_referenced=0,
            last_used=None,
        )

        self.db.add(memory)
        self.db.commit()

        logger.info(
            "confidence_memory_stored",
            memory_id=memory.id,
            final_confidence=final_confidence,
        )

        return str(memory.id)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update_confidence_from_reflection(
        self, reflection: Reflection, task_pattern: str
    ) -> None:
        """
        Nudge confidence scores up/down based on what the Reflection agent learned.

        FIX: Previously committed inside the loop — one commit per pattern.
        Now commits once after all updates (atomic, avoids partial writes).
        """
        if not reflection.confidence_updates:
            return

        logger.info(
            "confidence_updating", num_updates=len(reflection.confidence_updates)
        )

        for pattern, confidence_change in reflection.confidence_updates.items():
            memories = (
                self.db.query(Memory)
                .filter(Memory.task_pattern.like(f"%{pattern}%"))
                .order_by(desc(Memory.created_at))
                .limit(5)
                .all()
            )

            for memory in memories:
                old_confidence = (
                    float(memory.success_rate)
                    if memory.success_rate is not None
                    else 0.5
                )
                new_confidence = max(0.0, min(1.0, old_confidence + confidence_change))
                memory.success_rate = new_confidence

                logger.debug(
                    "confidence_updated",
                    memory_id=memory.id,
                    pattern=pattern,
                    old=old_confidence,
                    new=new_confidence,
                    change=confidence_change,
                )

        # FIX: single commit after all patterns are processed
        self.db.commit()

    # ------------------------------------------------------------------
    # Recency helper
    # ------------------------------------------------------------------

    def calculate_recency_score(self, memory: Memory) -> float:
        """
        Return a 0–1 recency score.  Decays exponentially with age:
        1.0 at 0 days → 0.5 at 30 days → ~0.1 at 90 days.

        FIX: The previous version used datetime.now(timezone.utc) (tz-aware)
        while SQLAlchemy DateTime columns without timezone=True store naive
        datetimes.  Subtracting aware from naive raises TypeError at runtime.
        Both sides now use naive UTC via datetime.utcnow().
        """
        if memory.last_used is None and memory.created_at is None:
            return 0.5

        last_time: datetime = memory.last_used or memory.created_at

        # Strip tzinfo if the value somehow arrives tz-aware (defensive)
        if last_time.tzinfo is not None:
            last_time = last_time.replace(tzinfo=None)

        now = datetime.utcnow()  # naive UTC — matches SQLAlchemy DateTime default
        days_old = max(0, (now - last_time).days)

        return max(0.1, 1.0 / (1.0 + days_old / 30.0))

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    async def recall_with_confidence(
        self,
        task_description: str,
        min_confidence: float = 0.3,
        limit: int = 3,
    ) -> Tuple[List[Dict], float]:
        """
        Return the most relevant past memories for the current task,
        together with their average confidence.

        Steps:
        1. Fetch candidates from DB (filtered by confidence threshold)
        2. Compute composite scores
        3. Ask LLM to pick the most relevant subset
        4. Update usage counters in a single batched DB query

        FIX: Previously ran one db.query() per LLM-selected ID (N+1 queries).
             Now batches all updates with a single IN filter.
        """
        logger.info("confidence_recall_starting", task=task_description)

        candidate_memories = (
            self.db.query(Memory)
            .filter(
                Memory.pattern_type == "success",
                Memory.success_rate >= min_confidence,
            )
            .order_by(
                desc(Memory.success_rate),
                desc(Memory.times_referenced),
                desc(Memory.created_at),
            )
            .limit(limit * 3)  # fetch extra so LLM can choose the best subset
            .all()
        )

        if not candidate_memories:
            logger.info("confidence_recall_empty")
            return [], 0.0

        # Build scored list
        scored_memories: List[Dict] = []
        for mem in candidate_memories:
            # FIX: explicit float() cast — DB drivers may return Decimal
            confidence = float(mem.success_rate) if mem.success_rate is not None else 0.0
            recency = self.calculate_recency_score(mem)
            usage = float(mem.times_referenced or 0)

            scored_memories.append(
                {
                    "id": mem.id,
                    "pattern": mem.task_pattern,
                    "task": mem.task_description,
                    "strategy": mem.strategy,
                    "tools": mem.tools_used,
                    "confidence": confidence,
                    "recency": recency,
                    "times_used": int(usage),
                    "composite_score": confidence * recency * (1 + usage**0.5 / 10),
                }
            )

        scored_memories.sort(key=lambda x: x["composite_score"], reverse=True)
        top_candidates = scored_memories[: limit * 2]

        # Ask the LLM which candidates are most relevant to the current task
        prompt = (
            f"Current task: {task_description}\n\n"
            f"Past successful patterns (with confidence scores):\n"
            f"{json.dumps(top_candidates, indent=2)}\n\n"
            f"Select the {limit} most relevant patterns for this task.\n"
            f"Consider both similarity to the current task and confidence score.\n"
            f'Return JSON only: {{"relevant_ids": ["id1", "id2", ...]}}'
        )

        try:
            response = await call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
            )

            response_text = response.strip()

            # Strip markdown fences if present
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1] if "\n" in response_text else response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

            # Extract JSON object
            start = response_text.find("{")
            end = response_text.rfind("}")
            if start != -1 and end != -1:
                response_text = response_text[start : end + 1]

            result = json.loads(response_text)
            relevant_ids: List[str] = result.get("relevant_ids", [])[:limit]

            # Build the return list from the already-scored data (no extra DB round-trip)
            id_to_scored = {m["id"]: m for m in scored_memories}
            relevant_memories = [
                id_to_scored[mid] for mid in relevant_ids if mid in id_to_scored
            ]
            confidences = [m["confidence"] for m in relevant_memories]

            # FIX: batch-update all matched Memory rows in one query instead of
            # one db.query() per ID (was N+1 queries).
            if relevant_ids:
                matched_objs = (
                    self.db.query(Memory)
                    .filter(Memory.id.in_(relevant_ids))
                    .all()
                )
                now = datetime.utcnow()
                for mem_obj in matched_objs:
                    mem_obj.times_referenced = int(mem_obj.times_referenced or 0) + 1
                    mem_obj.last_used = now
                self.db.commit()

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            logger.info(
                "confidence_recall_completed",
                num_memories=len(relevant_memories),
                avg_confidence=avg_confidence,
            )

            return relevant_memories, avg_confidence

        except Exception as e:
            logger.error("confidence_recall_error", error=str(e))
            fallback = scored_memories[:limit]
            
            # update usage counters even on fallback
            fallback_ids = [m["id"] for m in fallback]
            if fallback_ids:
                matched_objs = (
                    self.db.query(Memory)
                    .filter(Memory.id.in_(fallback_ids))
                    .all()
                )
                now = datetime.utcnow()
                for mem_obj in matched_objs:
                    mem_obj.times_referenced = int(mem_obj.times_referenced or 0) + 1
                    mem_obj.last_used = now
                self.db.commit()
            
            avg_conf = (
                sum(m["confidence"] for m in fallback) / len(fallback)
                if fallback
                else 0.0
            )
            return fallback, avg_conf