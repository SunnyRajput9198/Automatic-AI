import json
import structlog
from typing import List, Dict
from pydantic import BaseModel
from app.utils.json_parser import extract_json
from app.utils.llm import call_groq_with_system
logger = structlog.get_logger()


class PlanStep(BaseModel):
    step: int
    instruction: str
    reasoning: str


class PlannerAgent:
    """
    Converts user tasks into concrete, executable steps for the orchestrator.

    GUARANTEES:
    - Maximum 10 steps per plan
    - Every step has a non-empty instruction
    - Steps are ordered by dependency
    - No vague or human-imagination steps

    METHODS:
    - plan()   : create fresh plan from user task
    - replan() : create new plan when a step fails, avoiding the error
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
COMPARISON QUERIES (important):
- "what is X and how different from Y" → ONE step: web_search for "X vs Y"
- "compare X and Y" → ONE step: web_search for "X Y comparison"
- "difference between X and Y" → ONE step: web_search for "X Y difference"
- NEVER break comparison queries into separate searches for X and Y
- ONE search covers both topics

MULTI-TOPIC QUERIES:
- "what is X and Y" → ONE web_search covering both
- "explain X with examples" → ONE web_search, ONE file_write max
- Keep plans to 2 steps maximum for simple research tasks:
  Step 1: web_search
  Step 2: file_write (optional)
REPLAN RULES (when a step failed):
- Never repeat the exact same approach that failed
- If python_executor failed → try file_write to save code instead
- If web_search failed → try web_fetch with a direct URL
- If file_read failed → try file_list first to verify file exists
- Always address the specific error in your new plan
RESPOND ONLY WITH JSON. NO MARKDOWN, NO EXPLANATIONS.
"""
    REPLAN_SYSTEM_PROMPT = SYSTEM_PROMPT + """

REPLAN RULES (when a step failed):
- Never repeat the exact same approach that failed
- If python_executor failed → try file_write to save code instead
- If web_search failed → try web_fetch with a direct URL
- If file_read failed → try file_list first to verify file exists
- Always address the specific error in your new plan
"""
    def __init__(self, model: str = "llama-3.1-8b-instant"):
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
        response=""
        try:
            response = await call_groq_with_system(
    system_prompt=self.SYSTEM_PROMPT,
    user_prompt=user_prompt,
    model=self.model,
    temperature=0.1,
)

            # ---- ROBUST JSON EXTRACTION ----
            plan_data = extract_json(response, context="planner")
            if not plan_data:
                raise json.JSONDecodeError("No valid JSON found", response or "", 0)
            steps = plan_data.get("steps", [])[:10]

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
            response = await call_groq_with_system(
    system_prompt=self.REPLAN_SYSTEM_PROMPT,
    user_prompt=user_prompt,
    model=self.model,
    temperature=0.1,
)

            plan_data = extract_json(response, context="replanner")
            if not plan_data:
                raise json.JSONDecodeError("No valid JSON found", response or "", 0)
            steps = plan_data.get("steps", [])[:10] # enforce max 10 steps
            validated_steps: List[Dict] = []
            for idx, step in enumerate(steps, start=1):
                instruction = step.get("instruction")
                if not instruction:
                    logger.warning("planner_replan_invalid_step", step=step)
                    continue
                validated_steps.append({
                    "step": step.get("step", idx),
                    "instruction": instruction,
                    "reasoning": step.get("reasoning", ""),
                })

            if not validated_steps:
                raise ValueError("Replan produced no valid steps")

            logger.info("planner_replan_completed", num_steps=len(validated_steps))
            return validated_steps

        except Exception as e:
            logger.error("planner_replan_error", error=str(e))
            raise
