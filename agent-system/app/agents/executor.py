import json
import time
import structlog
from typing import Dict, Any, Optional
from app.utils.json_parser import extract_json
from app.utils.llm import call_llm_with_system, call_openai_with_system, call_openai_with_tools
from app.utils.file_manager import FileManager
from app.tools.base import Tool, ToolResult
from app.tools.python_tool import RestrictedPythonExecutor
from app.tools.shell_tool import ShellExecutor
from app.tools.file_tools import (
    FileReadTool,
    FileWriteTool,
    FileListTool,
    FileDeleteTool,
    FileAppendTool,
)
from app.tools.web_search import (
    WebSearchTool,
    WebFetchTool,
    SemanticScholarTool,
    WikipediaTool,
)
from app.tools.news_tool import NewsSearchTool
from app.core.config import settings
from app.utils.cost_tracker import global_cost_tracker

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

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self.tools: Dict[str, Tool] = {}

        self.file_manager = FileManager(base_dir=settings.WORKSPACE_DIR)

        if settings.ENABLE_PYTHON_EXECUTOR:
            self._register_tool(RestrictedPythonExecutor())

        if settings.ENABLE_SHELL:
            self._register_tool(ShellExecutor())

        # File and web tools are always available
        self._register_tool(FileReadTool(self.file_manager))
        self._register_tool(FileWriteTool(self.file_manager))
        self._register_tool(FileListTool(self.file_manager))
        self._register_tool(FileDeleteTool(self.file_manager))
        self._register_tool(FileAppendTool(self.file_manager))
        self._register_tool(WebSearchTool())
        self._register_tool(WebFetchTool())
        self._register_tool(SemanticScholarTool())
        self._register_tool(WikipediaTool())
        self._register_tool(NewsSearchTool())

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
        preferred_tools = context.get("preferred_tools", [])

        logger.info("executor_preferred_tools", tools=preferred_tools)

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
            if "session_history" in context and not any(
                tool_word in instruction_l
                for tool_word in [
                    "web_search",
                    "web_fetch",
                    "news_search",
                    "file_read",
                    "file_write",
                    "file_list",
                    "python_executor",
                    "shell_executor",
                ]
            ):
                history = context["session_history"]

                history_text = "\n\n".join(
                    h.get("output", "")[:1000] for h in history
                )

                prompt = f"""User question:
{instruction}

Session history:
{history_text}

Answer the user's question using the session history.
Do not repeat the raw history.
Provide a concise answer.
Use bullet points if appropriate.
"""

                logger.info("session_history_llm_prompt", prompt=prompt[:2000])

                response = await call_openai_with_system(
                    system_prompt=(
                        "You answer questions using session history. "
                        "Do not repeat raw history. "
                        "Provide concise answers. "
                        "Use bullet points for findings."
                    ),
                    user_prompt=prompt,
                )

                return ToolResult(
                    success=True,
                    output=response,
                    metadata={"source": "session_history"},
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
            t0 = time.time()
            result = await self.tools[tool_name].run(**tool_inputs)
            duration_ms = (time.time() - t0) * 1000

            global_cost_tracker.record_tool_call(
                tool_name=tool_name,
                agent="executor",
                success=result.success,
                duration_ms=duration_ms,
            )
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
    # Private: LLM-based tool selection via real tool binding
    # ------------------------------------------------------------------

    def _build_tool_schemas(self, avoid_tools: list) -> list:
        """Convert registered tools to OpenAI function schemas, excluding avoided tools."""
        return [
            tool.to_openai_schema()
            for name, tool in self.tools.items()
            if name not in avoid_tools
        ]

    async def _choose_tool(
        self, instruction: str, context: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Use OpenAI tool binding to pick the right tool and generate typed inputs.

        The LLM receives tool schemas via the `tools=` API parameter and responds
        with a structured `tool_calls` object — no free-text JSON parsing needed.

        Falls back to the old extract_json path if the model returns plain text
        (e.g. when forced_tool is set or the model declines to call a tool).
        """
        avoid_tools    = context.get("avoid_tools", [])
        forced_tool    = context.get("forced_tool")
        preferred_tools = context.get("preferred_tools", [])

        # Build tool schemas for binding (exclude avoided tools)
        tool_schemas = self._build_tool_schemas(avoid_tools)

        # Build context string for the user prompt
        context_str = ""
        if context:
            safe_context = {
                k: v for k, v in context.items()
                if k in [
                    "task_description", "should_search", "avoid_tools",
                    "forced_tool", "step_1_output", "step_2_output",
                    "step_3_output", "week4_output",
                ]
            }
            if "session_history" in context:
                safe_context["session_history"] = str(context["session_history"])[:300]
            context_str = "\n\nCONTEXT FROM PREVIOUS STEPS:\n" + json.dumps(
                safe_context, indent=2, default=str
            )

        preferred_tools_text = ""
        if preferred_tools:
            preferred_tools_text = (
                f"Historically successful tools: {', '.join(preferred_tools)}\n"
                "Prefer these tools when appropriate.\n\n"
            )

        system_prompt = (
            "You are a precise tool execution agent. "
            "Select the most appropriate tool for the given instruction and provide "
            "the exact executable inputs required.\n\n"
            "TOOL SELECTION GUIDE:\n"
            "- semantic_scholar_search → research papers, ML models, algorithms, academic topics\n"
            "- wikipedia_search        → factual lookups, definitions, general knowledge\n"
            "- news_search             → latest news, current events, today's headlines, breaking news\n"
            "- web_search              → ambiguous or broad queries needing multiple sources\n"
            "- web_fetch               → fetch content from a specific known URL\n"
            "- python_executor         → run executable Python code (provide actual code, not description)\n"
            "- shell_executor          → whitelisted shell commands\n"
            "- file_read/write/list/delete → workspace file operations\n\n"
            "IMPORTANT: For python_executor the 'code' input must be actual executable Python, "
            "not a description of what to do."
        )

        user_prompt = (
            f"{preferred_tools_text}"
            f"INSTRUCTION:\n{instruction}"
            f"{context_str}"
        )

        if forced_tool:
            user_prompt += f"\n\nYOU MUST USE TOOL: {forced_tool}"
        if avoid_tools:
            user_prompt += f"\n\nDO NOT USE THESE TOOLS: {avoid_tools}"

        try:
            result = await call_openai_with_tools(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tool_schemas,
                model=self.model,
                temperature=0.1,
                max_tokens=4000,
            )

            if result["type"] == "tool_call":
                # Structured response — tool name and typed arguments guaranteed
                logger.info(
                    "executor_tool_binding_success",
                    tool=result["name"],
                    args_keys=list(result["arguments"].keys()),
                )
                return {
                    "tool":      result["name"],
                    "inputs":    result["arguments"],
                    "reasoning": "selected via tool binding",
                }

            # Model returned plain text — fall back to JSON extraction
            logger.warning(
                "executor_tool_binding_text_fallback",
                preview=result["content"][:200],
            )
            decision = extract_json(result["content"], context="executor")
            if decision:
                return decision

            logger.error("executor_tool_binding_fallback_failed")
            return None

        except Exception as e:
            logger.error("executor_choice_error", error=str(e))
            return None
