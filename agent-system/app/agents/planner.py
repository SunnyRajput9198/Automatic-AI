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
2. Steps must be executable by tools (Python code, shell commands, web operations)
3. No vague instructions like "understand deeply" or "analyze thoroughly"
4. Each step should have clear inputs and outputs
5. Order steps logically (dependencies first)
6. Maximum 10 steps per task
7. Keep plans simple - avoid unnecessary parsing or extraction steps

AVAILABLE TOOLS:

📁 FILE OPERATIONS (persistent workspace):
- file_read: Read content from a file in the workspace
- file_write: Write content to a file in the workspace
- file_list: List all files in the workspace
- file_delete: Delete a file from the workspace

🌐 WEB OPERATIONS:
- web_search: Search the internet for information
  * Returns formatted results with titles, snippets, and URLs
  * NO parsing needed - output is ready to use
  * Use for: "search for X", "find tutorials", "what is X", "look up X"
- web_fetch: Fetch content from a specific URL
  * Use when you have a specific URL to retrieve

🐍 CODE EXECUTION:
- python_executor: Run Python code in a sandbox
  * Use for calculations, data processing, algorithms
  * Can import standard libraries
- shell_executor: Run allowed shell commands (ls, cat, echo, etc.)
  * Limited command whitelist for security
  * Use only when file_* tools don't apply

TOOL SELECTION GUIDELINES:

When user says "search for [topic]" or "find [topic]":
→ Use web_search (this means search the INTERNET, not files)

When user says "create file" or "read file" or mentions specific filenames:
→ Use file_read, file_write, file_list, or file_delete

When user wants calculations or data processing:
→ Use python_executor

When user explicitly mentions shell commands:
→ Use shell_executor (but prefer file_* tools when possible)

IMPORTANT NOTES:
- web_search returns PRE-FORMATTED results - no parsing step needed
- Avoid creating "parse results" or "extract data" steps after web_search
- Files created with file_write persist across tasks
- Python code in python_executor runs in a temporary sandbox

RESPONSE FORMAT (JSON only):
{
  "steps": [
    {
      "step": 1,
      "instruction": "Use web_search to find Python programming tutorials",
      "reasoning": "User wants to search for online tutorials"
    },
    {
      "step": 2,
      "instruction": "Save the top 3 results to a file called tutorials.txt using file_write",
      "reasoning": "Persist the results for later reference"
    }
  ]
}

BAD EXAMPLES:
❌ "Deeply understand the codebase"
❌ "Parse web search results" (web_search already returns formatted data!)
❌ "Extract URLs from search output" (URLs are already in the output!)
❌ "Search the filesystem for Python" (when user meant search the web)

GOOD EXAMPLES:
✅ "Use web_search to find React tutorials"
✅ "Use file_read to read config.json"
✅ "Use python_executor to calculate fibonacci(100)"
✅ "Use web_fetch to get content from https://example.com"
✅ "Use file_write to save results to output.txt"

DEFAULT INTERPRETATION:
- "search" = web_search (unless clearly about files)
- "find" = web_search (unless clearly about files)
- "what is" = web_search
- "look up" = web_search

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
