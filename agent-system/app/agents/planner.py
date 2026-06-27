import structlog
from typing import List, Dict
from app.utils.json_parser import extract_json
from app.utils.llm import call_openai_with_system

logger = structlog.get_logger()


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
IMPORTANT:

Do NOT use SESSION HISTORY unless the user explicitly asks about:
- previous task
- earlier research
- prior findings
- what was discussed before

For all other tasks, ignore SESSION HISTORY and plan normally.
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
✅ "Extract findings from SESSION HISTORY and answer the user's question"
✅ "Summarize key results contained in SESSION HISTORY"

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

SESSION HISTORY contains summaries of previous tasks.

IMPORTANT:
- SESSION HISTORY is already available in the task description.
- SESSION HISTORY is NOT a tool.
- SESSION HISTORY is NOT a file.
- Never create steps such as:
    * Use SESSION HISTORY
    * Read SESSION HISTORY
    * Query SESSION HISTORY
    * Retrieve SESSION HISTORY
- Never use file_read to access SESSION HISTORY.
- Never assume files like:
    * session_history.txt
    * session_history.json
    * previous_task.txt
    exist unless explicitly mentioned.

If the user asks:
- "what was the previous task"
- "what did you find previously"
- "summarize previous task"
- "summarize findings from previous task"
- "what were the findings"

Then use the SESSION HISTORY already provided in the task description.

Do NOT repeat or dump SESSION HISTORY. Instead:
- Extract relevant facts
- Summarize findings
- Answer the user's question
- Format results clearly (bullet points when appropriate)

CRITICAL: MEMORY QUERIES MUST PRODUCE EXACTLY ONE STEP.

For any request about previous task / findings / research / session history / conclusions / summaries:
Return exactly ONE step that directly answers the question.

GOOD:
Step 1: Answer the user's question using information from SESSION HISTORY.

GOOD:
Step 1: Summarize the key findings contained in SESSION HISTORY.

BAD:
Step 1: Extract findings
Step 2: Format findings

FILE NAMING RULES:
- Always use .txt extension for saving research results
- Never use .pdf, .docx, .xlsx — file_write only supports plain text
- Good: results.txt, research.txt, summary.txt
- Bad: results.pdf, paper.docx

REPLAN RULES (when a step failed):
- Never repeat the exact same approach that failed
- If python_executor failed → try file_write to save code instead
- If web_search failed → try web_fetch with a direct URL
- If file_read failed → try file_list first to verify file exists
- Always address the specific error in your new plan

RESPOND ONLY WITH JSON. NO MARKDOWN, NO EXPLANATIONS.
"""

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model

    def _validate_steps(self, steps: list, log_key: str) -> List[Dict]:
        """Validate and normalize raw step dicts from the LLM."""
        validated: List[Dict] = []
        for idx, step in enumerate(steps, start=1):
            instruction = step.get("instruction")
            if not instruction:
                logger.warning(f"{log_key}_invalid_step", step=step)
                continue
            validated.append(
                {
                    "step": step.get("step", idx),
                    "instruction": instruction,
                    "reasoning": step.get("reasoning", ""),
                }
            )
        return validated

    async def _call_llm(
        self, system_prompt: str, user_prompt: str, log_context: str
    ) -> List[Dict]:
        """
        Shared LLM call + parse + validate logic used by both plan() and replan().
        Raises ValueError on parse/validation failure.
        """
        response = await call_openai_with_system(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            temperature=0.1,
        )

        plan_data = extract_json(response, context=log_context)
        if not plan_data:
            logger.error(f"{log_context}_json_error",
                         error="No valid JSON found", response=response[:500])
            raise ValueError(f"Failed to parse {log_context} JSON: No valid JSON found")

        steps = plan_data.get("steps", [])[:10]
        validated = self._validate_steps(steps, log_context)
        if not validated:
            raise ValueError(f"{log_context} produced no valid steps")
        return validated

    async def plan(self, user_task: str) -> List[Dict]:
        """Convert user task into executable steps."""
        logger.info("planner_starting", task=user_task)
        user_prompt = f"USER TASK:\n{user_task}\n\nBreak this down into concrete, executable steps.\nReturn JSON only.\n"
        try:
            steps = await self._call_llm(self.SYSTEM_PROMPT, user_prompt, "planner")
            logger.info("planner_completed", num_steps=len(steps))
            return steps
        except ValueError:
            raise
        except Exception as e:
            logger.error("planner_error", error=str(e))
            raise

    async def replan(self, original_task: str, failed_step: str, error: str) -> List[Dict]:
        """Create a new plan when a step fails."""
        logger.info("planner_replanning", failed_step=failed_step)
        user_prompt = (
            f"ORIGINAL TASK:\n{original_task}\n\n"
            f"FAILED STEP:\n{failed_step}\n\n"
            f"ERROR:\n{error}\n\n"
            "The previous approach failed.\nCreate a NEW plan that avoids this error.\nReturn JSON only.\n"
        )
        try:
            steps = await self._call_llm(self.SYSTEM_PROMPT, user_prompt, "replanner")
            logger.info("planner_replan_completed", num_steps=len(steps))
            return steps
        except ValueError:
            raise
        except Exception as e:
            logger.error("planner_replan_error", error=str(e))
            raise
