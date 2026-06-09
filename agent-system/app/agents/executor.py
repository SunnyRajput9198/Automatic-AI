import json
import structlog
from typing import Dict, Any, Optional
from app.utils.json_parser import extract_json
from app.utils.llm import call_llm, call_llm_with_system
from app.utils.file_manager import FileManager
from app.tools.base import Tool, ToolResult
from app.tools.python_tool import PythonExecutor
from app.tools.shell_tool import ShellExecutor
from app.tools.file_tools import (
    FileReadTool,
    FileWriteTool,
    FileListTool,
    FileDeleteTool,
)
from app.tools.web_search import WebSearchTool, WebFetchTool
from app.core.config import settings

# This file answers: "Which tool should I use, and what exactly should I pass to it?"
logger = structlog.get_logger()


class ExecutorAgent:
    """
    Executor agent: picks the right tool for each step and runs it.

    Tools available:
    - python_executor  : run Python code in a sandbox
    - shell_executor   : run whitelisted shell commands (optional, disabled in prod)
    - file_read/write/list/delete : persistent workspace file operations
    - web_search / web_fetch      : internet access
    """

    SYSTEM_PROMPT = """You are a precise tool execution agent. Your job is to:
1. Read the step instruction
2. Choose the RIGHT tool
3. Generate the EXACT EXECUTABLE inputs needed

CRITICAL: For python_executor, you MUST provide actual executable Python code, NOT the instruction text!

AVAILABLE TOOLS:
{tools_description}

IMPORTANT NOTES ABOUT web_search:
- web_search ALREADY returns formatted, structured results
- Output includes: numbered results, titles, descriptions, and URLs
- NO parsing or extraction needed — results are ready to use

IMPORTANT:
If a task requires external libraries (flask, fastapi, django, etc.),
DO NOT execute them.
Instead, generate the code as a file using file_write without running it.

For code in JSON strings:
- Use escaped newlines: \\n
- Keep code simple and focused
- Avoid embedding large data structures in code

RESPONSE FORMAT (JSON only):
{
    "tool": "tool_name",
    "inputs": {
    "param1": "value1"
    },
    "reasoning": "why this tool and these inputs"
}

RESPOND ONLY WITH VALID JSON.
"""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self.tools: Dict[str, Tool] = {}

        self.file_manager = FileManager()

        if settings.ENABLE_PYTHON_EXECUTOR:
            self._register_tool(PythonExecutor())

        if settings.ENABLE_SHELL:
            self._register_tool(ShellExecutor())

        # File and web tools are always available
        self._register_tool(FileReadTool(self.file_manager))
        self._register_tool(FileWriteTool(self.file_manager))
        self._register_tool(FileListTool(self.file_manager))
        self._register_tool(FileDeleteTool(self.file_manager))
        self._register_tool(WebSearchTool())
        self._register_tool(WebFetchTool())

    def _register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        logger.info("tool_registered", tool=tool.name)

    def _get_tools_description(self) -> str:
        descriptions = []
        for tool in self.tools.values():
            extra = ""
            if tool.name == "python_executor":
                extra = "- The 'code' parameter must be EXECUTABLE Python code, not a description"
            elif tool.name == "shell_executor":
                extra = "- The 'command' parameter must be a valid shell command, not a description"
            elif tool.name.startswith("file_"):
                extra = "- Files persist across tasks in the shared workspace"

            descriptions.append(
                f"Tool: {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Input Schema: {json.dumps(tool.input_schema, indent=2)}\n"
                + (f"IMPORTANT for {tool.name}:\n{extra}\n" if extra else "")
            )
        return "\n".join(descriptions)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute_step(
        self, instruction: str, context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Choose a tool and execute a single plan step."""

        logger.info("executor_starting", instruction=instruction)
        context = context or {}
        avoid_tools = context.get("avoid_tools", [])

        # If python_executor is on the avoid list, force file_write instead
        if "python_executor" in avoid_tools:
            logger.warning("forcing_tool_due_to_failures", tool="file_write")
            context["forced_tool"] = "file_write"

        # ------------------------------------------------------------------
        # Deterministic fast-path: no LLM call needed for trivial cases
        # ------------------------------------------------------------------
        instruction_l = instruction.lower()
    
        logger.info("executor_fast_path_check", instruction_l=instruction_l[:100])
        logger.info("executor_context_keys", keys=list(context.keys()))
        try:
            # Fast-path: pure reasoning/summarization steps
            reasoning_keywords = [
                "summarize",
                "summary",
                "from the provided session history",
                "from session history",
                "from the session history",
                "previous task",
                "what was found",
                "based on session",
            ]
            if any(kw in instruction_l for kw in reasoning_keywords):
                if "session_history" in context:
                    history = context["session_history"]

                    output = []

                    for i, h in enumerate(history, 1):
                        output.append(f"Task {i}: {h.get('task', '')}")

                        if h.get("output"):
                            output.append(h["output"][:1000])

                    return ToolResult(
                        success=True,
                        output="\n\n".join(output),
                        metadata={"source": "session_history"}
                    )
        except Exception as e:
            logger.warning("executor_session_fast_path_failed", error=str(e))

        try:
            if instruction_l.startswith("list") or "list files" in instruction_l:
                if "workspace" in instruction_l or "persistent" in instruction_l:
                    return await self.tools["file_list"].run()
                elif settings.ENABLE_SHELL and "shell_executor" in self.tools:
                    return await self.tools["shell_executor"].run(command="ls -la")
                else:
                    return ToolResult(
                        success=False,
                        output="",
                        error="Shell executor is disabled; use file_list for workspace files.",
                    )
        except Exception as e:
            logger.warning(
                "executor_fallback_failed", instruction=instruction, error=str(e)
            )

        # ------------------------------------------------------------------
        # LLM-based tool selection
        # ------------------------------------------------------------------
        tool_decision = await self._choose_tool(instruction, context)

        if not tool_decision:
            return ToolResult(
                success=False,
                output="",
                error="Failed to choose appropriate tool",
            )

        tool_name = tool_decision.get("tool")
        tool_inputs = tool_decision.get("inputs", {})
        reasoning = tool_decision.get("reasoning", "")

        # Honour forced_tool override (set earlier when a tool is on avoid list)
        forced_tool = context.get("forced_tool")
        if forced_tool:
            logger.warning(
                "forcing_tool_override", chosen=tool_name, forced=forced_tool
            )
            tool_name = forced_tool

        logger.info("executor_tool_selected", tool=tool_name, reasoning=reasoning)

        # Hard block for avoided tools (catches edge cases where LLM still picked one)
        if tool_name in avoid_tools:
            logger.warning("blocked_avoided_tool", tool=tool_name)
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' is blocked due to repeated failures.",
            )

        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        # Extra validation for python_executor inputs
        if tool_name == "python_executor":
            code = tool_inputs.get("code", "")
            if not code:
                return ToolResult(
                    success=False,
                    output="",
                    error="python_executor requires a 'code' parameter.",
                )
            if code.lower().startswith(
                ("create a", "write a", "make a", "build a", "generate a", "produce a")
            ):
                logger.warning("executor_invalid_code_input", code_preview=code[:100])
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        "Received an instruction string instead of executable code. "
                        "Please provide actual Python code."
                    ),
                )

        # Run the tool
        try:
            result = await self.tools[tool_name].run(**tool_inputs)
            logger.info("executor_completed", tool=tool_name, success=result.success)
            return result
        except Exception as e:
            logger.error("executor_error", tool=tool_name, error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution failed: {str(e)}",
            )

    # ------------------------------------------------------------------
    # Private: LLM-based tool selection
    # ------------------------------------------------------------------

    async def _choose_tool(
        self, instruction: str, context: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Ask the LLM to pick the right tool and generate its inputs.
        Returns the parsed JSON decision dict, or None on failure.

        FIX: This was previously nested *inside* execute_step as an inner
        function, making it unreachable as a proper class method. The
        try/except block containing the actual call_llm call was also
        indented outside both methods entirely.
        """
        tools_desc = self._get_tools_description()
        system_prompt = self.SYSTEM_PROMPT.replace("{tools_description}", tools_desc)

        context_str = ""
        if context:
            # Limit context size to prevent recursion issues
            safe_context = {
                k: v
                for k, v in context.items()
                if k
                in [
                    "task_description",
                    "should_search",
                    "avoid_tools",
                    "forced_tool",
                    "step_1_output",
                    "step_2_output",
                    "step_3_output",
                    "week4_output",
                ]
            }
            # Add session history summary if present
            if "session_history" in context:
                safe_context["session_history"] = str(context["session_history"])[:300]
            context_str = "\n\nCONTEXT FROM PREVIOUS STEPS:\n" + json.dumps(
                safe_context, indent=2
            )
        user_prompt = (
            f"STEP INSTRUCTION:\n\n{instruction}{context_str}\n\n"
            "Choose the appropriate tool and generate EXECUTABLE inputs (not instruction text).\n"
            "For python_executor, generate actual Python code.\n"
            "For shell_executor, generate actual shell commands.\n"
            "For file_* tools, use appropriate filenames and content.\n"
            "For web_* tools, use proper queries or URLs.\n"
            "Return JSON only."
        )

        if context.get("forced_tool"):
            user_prompt += f"\n\nYOU MUST USE TOOL: {context['forced_tool']}"

        if context.get("avoid_tools"):
            user_prompt += f"\n\nDO NOT USE THESE TOOLS: {context['avoid_tools']}"

        try:
            response = await call_llm_with_system(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=0.1,
                max_tokens=4000,
            )

            response_text = response.strip()

            # Strip markdown code fences if the model wrapped the JSON
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                lines = lines[1:]  # drop the opening ```json / ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            # Attempt 1: direct parse
            decision = extract_json(response_text, context="executor")
            if not decision:
                logger.error(
                    "executor_json_extraction_failed",
                    response_preview=response_text[:200],
                )
                return None

            logger.debug(
                "executor_tool_decision",
                tool=decision.get("tool"),
                inputs_preview={
                    k: str(v)[:100] for k, v in decision.get("inputs", {}).items()
                },
            )
            return decision

        except Exception as e:
            logger.error("executor_choice_error", error=str(e))
            return None
