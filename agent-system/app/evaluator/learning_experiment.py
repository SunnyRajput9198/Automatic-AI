import asyncio
import json
import structlog
from pathlib import Path
from typing import List, Dict, Any
import uuid
from datetime import datetime, timezone

from app.db.session import get_db_context
from app.models.task import Task
from app.utils.cost_tracker import global_cost_tracker

logger = structlog.get_logger()


class LearningExperiment:
    """
    Runs the same task multiple times and measures system improvement across runs.

    Tracks:
    - Duration per run      — is the system getting faster?
    - Retry count           — is it failing less?
    - LLM calls             — is it being more efficient?
    - Search operations     — is it searching less?

    Usage:
        evaluator = LearningExperiment()
        await evaluator.run_learning_experiment(
            task_description="your task",
            num_runs=3,
            orchestrator_func=execute_task_v3,
        )
    """

    def __init__(self):
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)

    async def run_learning_experiment(
        self,
        task_description: str,
        num_runs: int = 3,
        orchestrator_func=None,
    ) -> Dict[str, Any]:
        """
        Run the same task num_runs times and return aggregated metrics.

        Args:
            task_description: Task prompt to repeat.
            num_runs:          How many iterations to run.
            orchestrator_func: Async callable that accepts a task_id string.

        Returns:
            Dict with per-run results and first→last improvement metrics.
        """
        logger.info("learning_experiment_starting", task=task_description, runs=num_runs)

        results: Dict[str, Any] = {
            "experiment": "learning",
            "task": task_description,
            "num_runs": num_runs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "runs": [],
        }

        for run_num in range(1, num_runs + 1):
            logger.info("experiment_run_starting", run=run_num)

            with get_db_context() as db:
                task = Task(
                    user_input=task_description,
                    status="PENDING",
                    id=str(uuid.uuid4()),
                )
                db.add(task)
                db.commit()
                task_id = task.id

            if orchestrator_func:
                try:
                    await orchestrator_func(task_id)
                except Exception as e:
                    logger.error("experiment_run_failed", run=run_num, error=str(e))

            with get_db_context() as db:
                task = db.query(Task).filter(Task.id == task_id).first()
                if not task:
                    continue

                duration = 0.0
                if task.completed_at and task.created_at:
                    duration = (task.completed_at - task.created_at).total_seconds()

                total_retries = sum(s.retry_count for s in task.steps)

                run_result: Dict[str, Any] = {
                    "run":          run_num,
                    "task_id":      task_id,
                    "status":       task.status,
                    "duration_sec": round(duration, 2),
                    "total_steps":  len(task.steps),
                    "total_retries": total_retries,
                    "success":      task.status == "COMPLETED",
                }

                completed_task = next(
                    (t for t in global_cost_tracker.completed_tasks if t.task_id == task_id),
                    None,
                )
                if completed_task:
                    run_result.update({
                        "llm_calls":        completed_task.total_llm_calls,
                        "reasoning_calls":  completed_task.reasoning_calls,
                        "search_operations": completed_task.search_operations,
                        "llm_efficiency":   completed_task.llm_efficiency,
                    })

                results["runs"].append(run_result)
                logger.info(
                    "experiment_run_completed",
                    run=run_num,
                    duration=run_result["duration_sec"],
                    retries=total_retries,
                )

            if run_num < num_runs:
                await asyncio.sleep(2)

        if not results["runs"]:
            logger.warning("experiment_no_runs_completed")
            results["improvement"] = {}
            return results

        if len(results["runs"]) >= 2:
            first = results["runs"][0]
            last  = results["runs"][-1]
            results["improvement"] = {
                "duration_reduction_sec": round(first["duration_sec"] - last["duration_sec"], 2),
                "duration_reduction_pct": round(
                    (first["duration_sec"] - last["duration_sec"]) / first["duration_sec"] * 100
                    if first["duration_sec"] > 0 else 0,
                    1,
                ),
                "retry_reduction":  max(0, first.get("total_retries", 0) - last.get("total_retries", 0)),
                "search_reduction": max(0, first.get("search_operations", 0) - last.get("search_operations", 0)),
                "llm_call_reduction": max(0, first.get("llm_calls", 0) - last.get("llm_calls", 0)),
            }

        self._save_experiment(results)
        self._print_learning_summary(results)
        return results

    def _save_experiment(self, results: Dict[str, Any]) -> None:
        """Persist experiment results as a timestamped JSON file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"experiment_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("experiment_saved", filename=str(filename))

    def _print_learning_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted table of per-run metrics and improvement deltas."""
        print("\n" + "=" * 80)
        print("  LEARNING EXPERIMENT RESULTS")
        print("=" * 80)
        print(f"\nTask: {results['task']}")
        print(f"Runs: {results['num_runs']}\n")

        # Build plain-text table without tabulate dependency
        headers = ["Run", "Status", "Duration(s)", "Steps", "Retries", "LLM Calls", "Searches"]
        col_w   = [5,     8,         12,            7,       9,         11,           9]

        header_row = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
        separator  = "  ".join("-" * w for w in col_w)
        print(header_row)
        print(separator)

        for run in results["runs"]:
            row = [
                str(run["run"]),
                "✓" if run["success"] else "✗",
                str(run["duration_sec"]),
                str(run["total_steps"]),
                str(run.get("total_retries", "N/A")),
                str(run.get("llm_calls", "N/A")),
                str(run.get("search_operations", "N/A")),
            ]
            print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))

        if "improvement" in results and results["improvement"]:
            imp = results["improvement"]
            print("\nIMPROVEMENTS (First → Last):")
            print(f"  Duration:   {imp['duration_reduction_pct']:+.1f}% ({imp['duration_reduction_sec']:+.1f}s)")
            print(f"  Retries:    {imp['retry_reduction']:+d}")
            print(f"  Searches:   {imp['search_reduction']:+d}")
            print(f"  LLM Calls:  {imp['llm_call_reduction']:+d}")

        print("\n" + "=" * 80 + "\n")
