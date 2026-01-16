import json
import structlog
from typing import List, Dict
from pydantic import BaseModel

from app.utils.llm import call_llm

logger = structlog.get_logger()


class PlanStep(BaseModel):
    step: int
    instruction: str
    reasoning: str


class PlannerAgent:
    """
    Converts high-level user tasks into concrete, executable steps.

    CRITICAL RULES:
    - Steps must be atomic (one clear action)
    - Steps must be tool-executable (no vague instructions)
    - No step should depend on human imagination
    - Each step must have clear success criteria
    """

    SYSTEM_PROMPT = """You are a precise task planning agent. Your job is to break down user requests into atomic, executable steps.

RULES:
1. Each step must be specific and actionable
2. Steps must be executable by tools (Python code, shell commands)
3. No vague instructions like "understand deeply" or "analyze thoroughly"
4. Each step should have clear inputs and outputs
5. Order steps logically (dependencies first)
6. Maximum 10 steps per task

AVAILABLE TOOLS:
- python_executor: Run Python code in a sandbox
- shell_executor: Run shell commands (ls, cat, grep, etc.)

RESPONSE FORMAT (JSON only):
{
  "steps": [
    {
      "step": 1,
      "instruction": "List all Python files in the current directory",
      "reasoning": "Need to identify what files exist before analyzing"
    }
  ]
}

BAD EXAMPLES:
❌ "Deeply understand the codebase"
❌ "Make the code better"
❌ "Think about potential issues"

GOOD EXAMPLES:
✅ "Run 'ls -la' to list all files with permissions"
✅ "Execute Python script test.py and capture output"
✅ "Search for TODO comments using grep"

RESPOND ONLY WITH JSON. NO MARKDOWN, NO EXPLANATIONS.
"""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        # Claude Sonnet is excellent for planning & decomposition
        self.model = model

    async def plan(self, user_task: str) -> List[Dict]:
        """
        Convert user task into executable steps
        """

        logger.info("planner_starting", task=user_task)

        user_prompt = f"""USER TASK:
{user_task}

Break this down into concrete, executable steps.
Return JSON only.
"""

        try:
            response = await call_llm(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.1,
            )

            # ---- ROBUST JSON EXTRACTION ----
            response_text = response.strip()
            start = response_text.find("{")
            end = response_text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise json.JSONDecodeError(
                    "No valid JSON object found", response_text, 0
                )

            plan_data = json.loads(response_text[start : end + 1])
            steps = plan_data.get("steps", [])

            logger.info("planner_completed", num_steps=len(steps))

            validated_steps: List[Dict] = []

            for idx, step in enumerate(steps, start=1):
                instruction = step.get("instruction")
                if not instruction:
                    logger.warning("planner_invalid_step", step=step)
                    continue

                validated_steps.append(
                    {
                        "step": step.get("step", idx),
                        "instruction": instruction,
                        "reasoning": step.get("reasoning", ""),
                    }
                )

            if not validated_steps:
                raise ValueError("No valid executable steps generated")

            return validated_steps

        except json.JSONDecodeError as e:
            logger.error(
                "planner_json_error",
                error=str(e),
                response=response,
            )
            raise ValueError(f"Failed to parse plan JSON: {str(e)}")

        except Exception as e:
            logger.error("planner_error", error=str(e))
            raise

    async def replan(
        self, original_task: str, failed_step: str, error: str
    ) -> List[Dict]:
        """
        Create a new plan when a step fails
        """

        logger.info("planner_replanning", failed_step=failed_step)

        user_prompt = f"""ORIGINAL TASK:
{original_task}

FAILED STEP:
{failed_step}

ERROR:
{error}

The previous approach failed.
Create a NEW plan that avoids this error.
Return JSON only.
"""

        try:
            response = await call_llm(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.3,
            )

            response_text = response.strip()
            start = response_text.find("{")
            end = response_text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise json.JSONDecodeError(
                    "No valid JSON object found", response_text, 0
                )

            plan_data = json.loads(response_text[start : end + 1])
            steps = plan_data.get("steps", [])

            logger.info("planner_replan_completed", num_steps=len(steps))
            return steps

        except Exception as e:
            logger.error("planner_replan_error", error=str(e))
            raise
