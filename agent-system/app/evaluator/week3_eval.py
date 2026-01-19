import asyncio
import json
import structlog
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tabulate import tabulate

from app.db.session import get_db_context
from app.models.task import Task
from app.utils.cost_tracker import global_cost_tracker

logger = structlog.get_logger()


class Week3Evaluator:
    """
    Evaluate Week 3 improvements through multi-run experiments.
    
    EXPERIMENT TYPES:
    1. Learning test: Same task 3x, measure improvement
    2. Confidence test: Varied tasks, track confidence evolution
    3. Search efficiency: Count searches over time
    4. Cost analysis: LLM calls, retries, duration
    """
    
    def __init__(self):
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
    
    async def run_learning_experiment(
        self,
        task_description: str,
        num_runs: int = 3,
        orchestrator_func = None  # Function to run orchestration
    ) -> Dict[str, Any]:
        """
        Run the same task multiple times and measure improvement.
        
        Args:
            task_description: Task to repeat
            num_runs: Number of iterations
            orchestrator_func: Async function to execute task
            
        Returns:
            Experiment results with metrics
        """
        logger.info(
            "learning_experiment_starting",
            task=task_description,
            runs=num_runs
        )
        
        results = {
            "experiment": "learning",
            "task": task_description,
            "num_runs": num_runs,
            "timestamp": datetime.utcnow().isoformat(),
            "runs": []
        }
        
        for run_num in range(1, num_runs + 1):
            logger.info("experiment_run_starting", run=run_num)
            
            # Create task in DB
            with get_db_context() as db:
                task = Task(
                    user_input=task_description,
                    status="PENDING"
                )
                db.add(task)
                db.commit()
                task_id = task.id
            
            # Run orchestrator if provided
            if orchestrator_func:
                try:
                    await orchestrator_func(task_id)
                except Exception as e:
                    logger.error("experiment_run_failed", run=run_num, error=str(e))
            
            # Collect metrics
            with get_db_context() as db:
                task = db.query(Task).filter(Task.id == task_id).first()
                
                if task:
                    duration = 0.0
                    if task.completed_at and task.created_at:
                        duration = (task.completed_at - task.created_at).total_seconds()
                    
                    total_retries = sum(step.retry_count for step in task.steps)
                    
                    run_result = {
                        "run": run_num,
                        "task_id": task_id,
                        "status": task.status,
                        "duration_sec": round(duration, 2),
                        "total_steps": len(task.steps),
                        "total_retries": total_retries,
                        "success": task.status == "COMPLETED"
                    }
                    
                    # Add cost tracker data if available
                    completed_task = next(
                        (t for t in global_cost_tracker.completed_tasks if t.task_id == task_id),
                        None
                    )
                    if completed_task:
                        run_result.update({
                            "llm_calls": completed_task.total_llm_calls,
                            "reasoning_calls": completed_task.reasoning_calls,
                            "search_operations": completed_task.search_operations,
                            "llm_efficiency": completed_task.llm_efficiency
                        })
                    
                    results["runs"].append(run_result)
                    
                    logger.info(
                        "experiment_run_completed",
                        run=run_num,
                        duration=run_result["duration_sec"],
                        retries=total_retries
                    )
            
            # Wait between runs to allow learning
            if run_num < num_runs:
                await asyncio.sleep(2)
        
        # Calculate improvements
        if len(results["runs"]) >= 2:
            first_run = results["runs"][0]
            last_run = results["runs"][-1]
            
            results["improvement"] = {
                "duration_reduction_sec": round(
                    first_run["duration_sec"] - last_run["duration_sec"],
                    2
                ),
                "duration_reduction_pct": round(
                    ((first_run["duration_sec"] - last_run["duration_sec"]) / first_run["duration_sec"] * 100)
                    if first_run["duration_sec"] > 0 else 0,
                    1
                ),
                "retry_reduction": first_run.get("total_retries", 0) - last_run.get("total_retries", 0),
                "search_reduction": first_run.get("search_operations", 0) - last_run.get("search_operations", 0),
                "llm_call_reduction": first_run.get("llm_calls", 0) - last_run.get("llm_calls", 0)
            }
        
        # Save results
        self._save_experiment(results)
        
        # Print summary
        self._print_learning_summary(results)
        
        return results
    
    def _save_experiment(self, results: Dict[str, Any]):
        """Save experiment results to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"experiment_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("experiment_saved", filename=str(filename))
    
    def _print_learning_summary(self, results: Dict[str, Any]):
        """Print a nice table showing learning progress"""
        print("\n" + "="*80)
        print("🧠 WEEK 3 LEARNING EXPERIMENT RESULTS")
        print("="*80)
        print(f"\nTask: {results['task']}")
        print(f"Runs: {results['num_runs']}\n")
        
        # Create table
        table_data = []
        headers = ["Run", "Status", "Duration (s)", "Steps", "Retries", "LLM Calls", "Searches"]
        
        for run in results["runs"]:
            table_data.append([
                run["run"],
                "✓" if run["success"] else "✗",
                run["duration_sec"],
                run["total_steps"],
                run.get("total_retries", "N/A"),
                run.get("llm_calls", "N/A"),
                run.get("search_operations", "N/A")
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print improvements
        if "improvement" in results:
            imp = results["improvement"]
            print("\n📈 IMPROVEMENTS (First → Last):")
            print(f"   Duration:    {imp['duration_reduction_pct']:+.1f}% ({imp['duration_reduction_sec']:+.1f}s)")
            print(f"   Retries:     {imp['retry_reduction']:+d}")
            print(f"   Searches:    {imp['search_reduction']:+d}")
            print(f"   LLM Calls:   {imp['llm_call_reduction']:+d}")
        
        print("\n" + "="*80 + "\n")
    
    async def quick_demo(self, orchestrator_func=None):
        """
        Quick demo showing Week 3 capabilities.
        
        Runs 3 test tasks:
        1. File operation (should not search)
        2. Calculation (should use memory after first run)
        3. Web research (should search)
        """
        print("\n🚀 WEEK 3 QUICK DEMO\n")
        
        test_tasks = [
            "Create a file called demo.txt with the text 'Hello Week 3'",
            "Calculate the factorial of 20 using Python",
            "Search for the latest Python 3.12 features"
        ]
        
        for task in test_tasks:
            print(f"\n📋 Testing: {task}")
            
            # Run twice to show learning
            await self.run_learning_experiment(
                task_description=task,
                num_runs=2,
                orchestrator_func=orchestrator_func
            )
            
            await asyncio.sleep(1)
        
        print("\n✅ Demo completed! Check evaluation_results/ for detailed data.\n")


# Global evaluator instance
week3_evaluator = Week3Evaluator()


# Standalone test function
async def test_week3_evaluation():
    """Test the evaluation framework without full orchestrator"""
    print("Testing Week 3 Evaluation Framework...")
    
    # Mock orchestrator for testing
    async def mock_orchestrator(task_id: str):
        await asyncio.sleep(0.5)  # Simulate work
        
        with get_db_context() as db:
            task = db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = "COMPLETED"
                task.completed_at = datetime.utcnow()
                db.commit()
    
    await week3_evaluator.run_learning_experiment(
        task_description="Test task for evaluation",
        num_runs=3,
        orchestrator_func=mock_orchestrator
    )


if __name__ == "__main__":
    asyncio.run(test_week3_evaluation())