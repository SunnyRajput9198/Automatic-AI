import time
import json
import os
import structlog
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field
from collections import defaultdict

logger = structlog.get_logger()

COSTS_DIR = os.getenv("COSTS_DIR", "/app/costs")

# ---------------------------------------------------------------------------
# Pricing table  (USD per 1M tokens, input+output blended estimate)
# Keep this as the single source of truth — add new models here as needed.
# ---------------------------------------------------------------------------
COST_PER_1M_TOKENS: Dict[str, float] = {
    "claude-haiku-4-5-20251001":   0.25,   # Claude Haiku (input ~$0.25, output ~$1.25)
    "claude-sonnet-4-5-20251001":  3.00,   # Claude Sonnet
    "claude-opus-4-5-20251001":   15.00,   # Claude Opus
    "gpt-5-mini":                   0.15,   # GPT-4o-mini equivalent
    "gpt-4o":                       5.00,
    "gpt-4o-mini":                  0.15,
}
_DEFAULT_COST_PER_1M = 1.00   # fallback for unknown models


@dataclass
class LLMCall:
    """Single LLM call record."""
    agent:             str
    model:             str
    duration_ms:       float
    tokens_estimated:  int
    purpose:           str
    timestamp:         float = field(default_factory=time.time)


@dataclass
class ToolCall:
    """Single tool execution record."""
    tool_name:   str
    agent:       str
    success:     bool
    duration_ms: float
    timestamp:   float = field(default_factory=time.time)


@dataclass
class TaskCost:
    """Complete cost breakdown for one task."""
    task_id:      str
    started_at:   float
    completed_at: Optional[float]

    # ── LLM records ──────────────────────────────────────────────────────
    llm_calls:         List[LLMCall]
    total_llm_calls:   int
    reasoning_calls:   int
    planning_calls:    int
    execution_calls:   int
    critic_calls:      int
    reflection_calls:  int

    # Per-agent LLM call counts  {agent_name: count}
    llm_calls_by_agent: Dict[str, int]

    # Per-agent estimated token sums  {agent_name: tokens}
    tokens_by_agent: Dict[str, int]

    # ── Tool records ─────────────────────────────────────────────────────
    tool_calls:          List[ToolCall]
    tool_calls_by_name:  Dict[str, int]    # {tool_name: call_count}
    tool_success_by_name: Dict[str, int]   # {tool_name: success_count}

    # ── Execution counts ─────────────────────────────────────────────────
    total_retries:    int
    total_steps:      int
    search_operations: int

    # ── Timing ───────────────────────────────────────────────────────────
    duration_sec: float

    # ── Outcome ──────────────────────────────────────────────────────────
    success: bool

    # ── Derived efficiency metrics ────────────────────────────────────────
    cost_per_step:  float   # llm_calls / steps
    llm_efficiency: float   # steps / llm_calls  (higher = fewer LLM calls per step)


class CostTracker:
    """
    Track and analyse costs for all agent operations.

    Tracks:
    - Per-agent LLM call counts and estimated token usage
    - Per-tool call counts and success rates
    - Estimated USD cost per model
    - Overall efficiency metrics
    """

    # ~4 chars per token (rough estimate for English prose)
    TOKENS_PER_CHAR = 0.25

    def __init__(self):
        self.current_task:    Optional[TaskCost] = None
        self.completed_tasks: List[TaskCost]     = []
        Path(COSTS_DIR).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def start_task(self, task_id: str):
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
            llm_calls_by_agent={},
            tokens_by_agent={},
            tool_calls=[],
            tool_calls_by_name={},
            tool_success_by_name={},
            total_retries=0,
            total_steps=0,
            search_operations=0,
            duration_sec=0.0,
            success=False,
            cost_per_step=0.0,
            llm_efficiency=0.0,
        )
        logger.info("cost_tracking_started", task_id=task_id)

    def complete_task(self, success: bool):
        if not self.current_task:
            logger.warning("cost_tracker_no_active_task")
            return

        t = self.current_task
        t.completed_at = time.time()
        t.duration_sec = round(t.completed_at - t.started_at, 2)
        t.success = success

        if t.total_steps > 0:
            t.cost_per_step  = round(t.total_llm_calls / t.total_steps, 2)
            t.llm_efficiency = round(t.total_steps / max(1, t.total_llm_calls), 2)

        self.completed_tasks.append(t)
        self._export_task_cost(t)

        logger.info(
            "cost_tracking_completed",
            task_id=t.task_id,
            duration_sec=t.duration_sec,
            total_llm_calls=t.total_llm_calls,
            llm_efficiency=t.llm_efficiency,
            llm_calls_by_agent=t.llm_calls_by_agent,
            tool_calls_by_name=t.tool_calls_by_name,
        )
        self.current_task = None

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        agent:           str,
        model:           str,
        response_length: int,
        purpose:         str,
        duration_ms:     float,
    ):
        """Record one LLM call with per-agent and per-purpose tracking."""
        if not self.current_task:
            return

        t = self.current_task
        tokens_est = int(response_length * self.TOKENS_PER_CHAR)

        t.llm_calls.append(LLMCall(
            agent=agent,
            model=model,
            duration_ms=duration_ms,
            tokens_estimated=tokens_est,
            purpose=purpose,
        ))
        t.total_llm_calls += 1

        # Per-purpose counters
        purpose_map = {
            "reasoning":  "reasoning_calls",
            "planning":   "planning_calls",
            "execution":  "execution_calls",
            "critic":     "critic_calls",
            "reflection": "reflection_calls",
        }
        attr = purpose_map.get(purpose)
        if attr:
            setattr(t, attr, getattr(t, attr) + 1)

        # Per-agent counters
        t.llm_calls_by_agent[agent] = t.llm_calls_by_agent.get(agent, 0) + 1
        t.tokens_by_agent[agent]    = t.tokens_by_agent.get(agent, 0) + tokens_est

        logger.debug(
            "llm_call_recorded",
            agent=agent,
            model=model,
            purpose=purpose,
            tokens=tokens_est,
            duration_ms=round(duration_ms, 1),
        )

    def record_tool_call(
        self,
        tool_name:   str,
        agent:       str,
        success:     bool,
        duration_ms: float = 0.0,
    ):
        """Record one tool execution with per-tool success tracking."""
        if not self.current_task:
            return

        t = self.current_task
        t.tool_calls.append(ToolCall(
            tool_name=tool_name,
            agent=agent,
            success=success,
            duration_ms=duration_ms,
        ))
        t.tool_calls_by_name[tool_name] = t.tool_calls_by_name.get(tool_name, 0) + 1
        if success:
            t.tool_success_by_name[tool_name] = (
                t.tool_success_by_name.get(tool_name, 0) + 1
            )

        logger.debug(
            "tool_call_recorded",
            tool=tool_name,
            agent=agent,
            success=success,
        )

    def record_retry(self):
        if self.current_task:
            self.current_task.total_retries += 1

    def record_step(self):
        if self.current_task:
            self.current_task.total_steps += 1

    def record_search(self):
        if self.current_task:
            self.current_task.search_operations += 1

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_task_cost(self, t: TaskCost):
        filename = f"{COSTS_DIR}/task_{t.task_id}.json"
        data = asdict(t)

        # ── Estimated monetary cost ───────────────────────────────────
        tokens_by_model: Dict[str, int] = defaultdict(int)
        for call in t.llm_calls:
            tokens_by_model[call.model] += call.tokens_estimated

        cost_by_model: Dict[str, float] = {}
        total_cost_usd = 0.0
        for model, tokens in tokens_by_model.items():
            rate = COST_PER_1M_TOKENS.get(model, _DEFAULT_COST_PER_1M)
            cost = round((tokens / 1_000_000) * rate, 6)
            cost_by_model[model] = cost
            total_cost_usd += cost

        data["total_tokens_estimated"] = sum(tokens_by_model.values())
        data["cost_by_model"]          = cost_by_model
        data["estimated_cost_usd"]     = round(total_cost_usd, 6)

        # ── Per-tool success rate ─────────────────────────────────────
        tool_success_rate: Dict[str, float] = {}
        for tool, total in t.tool_calls_by_name.items():
            successes = t.tool_success_by_name.get(tool, 0)
            tool_success_rate[tool] = round(successes / total, 2) if total else 0.0
        data["tool_success_rate"] = tool_success_rate

        # ── Most expensive agent ──────────────────────────────────────
        if t.tokens_by_agent:
            data["most_expensive_agent"] = max(
                t.tokens_by_agent, key=lambda k: t.tokens_by_agent[k]
            )

        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug("cost_export_saved", filename=filename)
        except Exception as e:
            logger.error("cost_export_failed", filename=filename, error=str(e))

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        if not self.completed_tasks:
            return {"total_tasks": 0, "success_rate": 0.0}

        total     = len(self.completed_tasks)
        successes = sum(1 for t in self.completed_tasks if t.success)

        # Aggregate per-agent token usage across all tasks
        agg_tokens: Dict[str, int] = defaultdict(int)
        agg_tools:  Dict[str, int] = defaultdict(int)
        for t in self.completed_tasks:
            for agent, tokens in t.tokens_by_agent.items():
                agg_tokens[agent] += tokens
            for tool, count in t.tool_calls_by_name.items():
                agg_tools[tool] += count

        return {
            "total_tasks":       total,
            "successes":         successes,
            "failures":          total - successes,
            "success_rate":      round(successes / total, 2),
            "avg_duration_sec":  round(
                sum(t.duration_sec for t in self.completed_tasks) / total, 2
            ),
            "avg_llm_calls":     round(
                sum(t.total_llm_calls for t in self.completed_tasks) / total, 2
            ),
            "avg_retries":       round(
                sum(t.total_retries for t in self.completed_tasks) / total, 2
            ),
            "avg_efficiency":    round(
                sum(t.llm_efficiency for t in self.completed_tasks) / total, 2
            ),
            "total_searches":    sum(t.search_operations for t in self.completed_tasks),
            "tokens_by_agent":   dict(agg_tokens),
            "tool_calls_by_name": dict(agg_tools),
        }

    def get_agent_breakdown(self) -> Dict[str, Any]:
        """Per-agent cost/call summary across all completed tasks."""
        agg: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"llm_calls": 0, "tokens": 0, "tasks": 0}
        )
        for t in self.completed_tasks:
            for agent, calls in t.llm_calls_by_agent.items():
                agg[agent]["llm_calls"] += calls
                agg[agent]["tokens"]    += t.tokens_by_agent.get(agent, 0)
                agg[agent]["tasks"]     += 1

        # Add estimated cost per agent
        for agent, stats in agg.items():
            # Use default rate since we don't store per-agent model here
            stats["estimated_cost_usd"] = round(
                (stats["tokens"] / 1_000_000) * _DEFAULT_COST_PER_1M, 6
            )

        return dict(agg)

    def get_tool_breakdown(self) -> Dict[str, Any]:
        """Per-tool call count and success rate across all completed tasks."""
        total_calls:   Dict[str, int] = defaultdict(int)
        total_success: Dict[str, int] = defaultdict(int)

        for t in self.completed_tasks:
            for tool, count in t.tool_calls_by_name.items():
                total_calls[tool]   += count
                total_success[tool] += t.tool_success_by_name.get(tool, 0)

        return {
            tool: {
                "calls":        total_calls[tool],
                "successes":    total_success[tool],
                "success_rate": round(total_success[tool] / total_calls[tool], 2)
                               if total_calls[tool] else 0.0,
            }
            for tool in total_calls
        }

    def compare_tasks(self, task_id_1: str, task_id_2: str) -> Dict[str, Any]:
        t1 = next((t for t in self.completed_tasks if t.task_id == task_id_1), None)
        t2 = next((t for t in self.completed_tasks if t.task_id == task_id_2), None)

        if not t1 or not t2:
            return {"error": "One or both tasks not found"}

        return {
            "task_1": task_id_1,
            "task_2": task_id_2,
            "duration_improvement_pct": round(
                ((t1.duration_sec - t2.duration_sec) / t1.duration_sec) * 100, 1
            ) if t1.duration_sec > 0 else 0,
            "llm_call_delta":   t1.total_llm_calls - t2.total_llm_calls,
            "retry_delta":      t1.total_retries - t2.total_retries,
            "efficiency_delta": round(t2.llm_efficiency - t1.llm_efficiency, 2),
        }


# Global singleton
global_cost_tracker = CostTracker()
