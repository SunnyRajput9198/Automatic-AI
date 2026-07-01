import re
import json
import time
import structlog
from typing import Dict, Any, Optional

from app.utils.json_parser import extract_json
from app.utils.llm import call_openai_with_system, call_openai_with_tools
from app.utils.file_manager import FileManager
from app.utils.cost_tracker import global_cost_tracker
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
from app.tools.weather_tool import WeatherTool
from app.core.config import settings

logger = structlog.get_logger()

# Context keys passed from loop_v3 that are useful to the LLM tool selector.
# step_N_output keys are matched by prefix below.
_SAFE_CONTEXT_KEYS = frozenset({
    "task_description",
    "should_search",
    "avoid_tools",
    "forced_tool",
    "week4_output",
})

# Tool names that should bypass the session-history fast-path
_TOOL_WORDS = frozenset({
    "web_search", "web_fetch", "news_search",
    "semantic_scholar_search", "wikipedia_search",
    "file_read", "file_write", "file_list", "file_append", "file_delete",
    "python_executor", "shell_executor",
})

# System prompt for the tool-binding LLM call
_TOOL_SELECTION_SYSTEM = (
    "You are a precise tool execution agent. "
    "Select the most appropriate tool for the given instruction and provide "
    "the exact executable inputs required.\n\n"
    "TOOL SELECTION GUIDE:\n"
    "- semantic_scholar_search → research papers, ML models, algorithms, academic topics\n"
    "- wikipedia_search        → factual lookups, definitions, general knowledge\n"
    "- news_search             → latest news, current events, today's headlines, breaking news\n"
    "- get_weather             → current temperature and weather for any city — ALWAYS use this for temperature/weather queries\n"
    "- web_search              → ambiguous or broad queries needing multiple sources\n"
    "- web_fetch               → fetch content from a specific known URL\n"
    "- news_search             → current news, today's headlines, breaking news, recent events\n"
    "- python_executor         → run executable Python code (provide actual code, not a description)\n"
    "- shell_executor          → whitelisted shell commands\n"
    "- file_read/write/append/list/delete → workspace file operations\n\n"
    "IMPORTANT: When a file_write step says to save results from previous steps,\n"
    "check the CONTEXT FROM PREVIOUS STEPS section — step_1_output, step_2_output etc.\n"
    "are available there. Use them to compose the file content directly.\n"
    "Do NOT do another search — just write the file with the already-gathered data.\n\n"
    "IMPORTANT: For python_executor the 'code' input must be actual executable Python, "
    "not a description of what to do."
)


class ExecutorAgent:
    """
    Picks the right tool for each plan step and runs it.

    Tool selection uses OpenAI tool binding — the LLM receives all registered
    tool schemas via the `tools=` parameter and returns a structured tool_call.
    Falls back to extract_json if the model returns plain text.
    """

    def __init__(self, model: str = "gpt-5-mini"):
        self.model        = model
        self.tools: Dict[str, Tool] = {}
        self.file_manager = FileManager(base_dir=settings.WORKSPACE_DIR)

        if settings.ENABLE_PYTHON_EXECUTOR:
            self._register(RestrictedPythonExecutor())
        if settings.ENABLE_SHELL:
            self._register(ShellExecutor())

        for tool in [
            FileReadTool(self.file_manager),
            FileWriteTool(self.file_manager),
            FileListTool(self.file_manager),
            FileDeleteTool(self.file_manager),
            FileAppendTool(self.file_manager),
            WebSearchTool(),
            WebFetchTool(),
            SemanticScholarTool(),
            WikipediaTool(),
            NewsSearchTool(),
            WeatherTool(),
        ]:
            self._register(tool)

    def _register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        logger.info("tool_registered", tool=tool.name)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute_step(
        self, instruction: str, context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Choose a tool via LLM tool binding and execute a single plan step."""
        logger.info("executor_starting", instruction=instruction[:120])
        context       = context or {}
        avoid_tools   = context.get("avoid_tools", [])
        instruction_l = instruction.lower()

        # Force file_write when python_executor is on the avoid list
        if "python_executor" in avoid_tools:
            logger.warning("forcing_tool_due_to_failures", tool="file_write")
            context["forced_tool"] = "file_write"

        # ── Fast-path 1: session-history summarisation (no tool needed) ──
        if "session_history" in context and not any(w in instruction_l for w in _TOOL_WORDS):
            try:
                history_text = "\n\n".join(
                    h.get("output", "")[:1000] for h in context["session_history"]
                )
                response = await call_openai_with_system(
                    system_prompt=(
                        "You answer questions using session history. "
                        "Do not repeat raw history. "
                        "Provide concise answers. "
                        "Use bullet points for findings."
                    ),
                    user_prompt=(
                        f"User question:\n{instruction}\n\n"
                        f"Session history:\n{history_text}\n\n"
                        "Answer concisely using the session history."
                    ),
                )
                return ToolResult(
                    success=True,
                    output=response,
                    metadata={"source": "session_history"},
                )
            except Exception as e:
                logger.warning("executor_session_fast_path_failed", error=str(e))

        # ── Fast-path 2: trivial file-list ───────────────────────────────
        if instruction_l.startswith("list") or "list files" in instruction_l:
            try:
                if "workspace" in instruction_l or "persistent" in instruction_l:
                    return await self.tools["file_list"].run()
                if settings.ENABLE_SHELL and "shell_executor" in self.tools:
                    return await self.tools["shell_executor"].run(command="ls -la")
                return ToolResult(
                    success=False, output="",
                    error="Shell executor disabled; use file_list for workspace files.",
                )
            except Exception as e:
                logger.warning("executor_list_fast_path_failed", error=str(e))

        # ── Fast-path 3: file_write summary step ─────────────────────────
        # When the instruction asks to save/write/combine previous step results
        # and step outputs exist in context, build the file content directly
        # rather than asking the LLM (which may search again instead of writing).
        is_write_step = (
            "file_write" in instruction_l
            or ("save" in instruction_l and ("findings" in instruction_l or "results" in instruction_l or "combined" in instruction_l))
            or ("write" in instruction_l and "results.txt" in instruction_l)
        )
        step_outputs = {k: v for k, v in context.items() if k.startswith("step_") and k.endswith("_output")}

        if is_write_step and step_outputs and "file_list" not in instruction_l:
            try:
                # Extract filename from instruction
                fname_match = re.search(r"[\w\-]+\.txt", instruction_l)
                filename = fname_match.group(0) if fname_match else "results.txt"

                # Build content from all step outputs
                parts = []
                for k in sorted(step_outputs.keys()):
                    step_num = k.replace("step_", "").replace("_output", "")
                    parts.append(f"=== Step {step_num} Results ===\n{step_outputs[k]}")
                content = "\n\n".join(parts)

                logger.info("executor_file_write_fast_path", filename=filename, steps=list(step_outputs.keys()))
                result = await self.tools["file_write"].run(filename=filename, content=content)
                if result.success:
                    return result
                # If write failed, fall through to LLM path
            except Exception as e:
                logger.warning("executor_file_write_fast_path_failed", error=str(e))
        # ── LLM tool selection ────────────────────────────────────────────
        tool_decision = await self._choose_tool(instruction, context)
        if not tool_decision:
            return ToolResult(success=False, output="", error="Failed to choose appropriate tool")

        tool_name   = tool_decision.get("tool")
        tool_inputs = tool_decision.get("inputs", {})
        reasoning   = tool_decision.get("reasoning", "")

        # Honour forced_tool override
        forced_tool = context.get("forced_tool")
        if forced_tool:
            logger.warning("forcing_tool_override", chosen=tool_name, forced=forced_tool)
            tool_name = forced_tool

        logger.info("executor_tool_selected", tool=tool_name, reasoning=reasoning)

        if tool_name in avoid_tools:
            logger.warning("blocked_avoided_tool", tool=tool_name)
            return ToolResult(
                success=False, output="",
                error=f"Tool '{tool_name}' is blocked due to repeated failures.",
            )

        if tool_name not in self.tools:
            return ToolResult(success=False, output="", error=f"Unknown tool: {tool_name}")

        # Validate python_executor inputs
        if tool_name == "python_executor":
            code = tool_inputs.get("code", "")
            if not code:
                return ToolResult(
                    success=False, output="",
                    error="python_executor requires a 'code' parameter.",
                )
            if code.lower().startswith(
                ("create a", "write a", "make a", "build a", "generate a", "produce a")
            ):
                logger.warning("executor_invalid_code_input", code_preview=code[:100])
                return ToolResult(
                    success=False, output="",
                    error=(
                        "Received an instruction string instead of executable code. "
                        "Please provide actual Python code."
                    ),
                )

        # Run the tool
        try:
            t0     = time.time()
            result = await self.tools[tool_name].run(**tool_inputs)
            global_cost_tracker.record_tool_call(
                tool_name=tool_name,
                agent="executor",
                success=result.success,
                duration_ms=(time.time() - t0) * 1000,
            )
            logger.info("executor_completed", tool=tool_name, success=result.success)
            return result
        except Exception as e:
            logger.error("executor_error", tool=tool_name, error=str(e))
            return ToolResult(success=False, output="", error=f"Tool execution failed: {str(e)}")

    # ------------------------------------------------------------------
    # Private: LLM tool binding
    # ------------------------------------------------------------------

    def _build_tool_schemas(self, avoid_tools: list) -> list:
        return [
            tool.to_openai_schema()
            for name, tool in self.tools.items()
            if name not in avoid_tools
        ]

    def _build_context_str(self, context: Dict[str, Any]) -> str:
        """
        Extract the subset of context that's useful for tool selection.
        Includes fixed keys + any step_N_output keys dynamically.
        Skips if the result would be an empty dict.
        """
        safe: Dict[str, Any] = {}

        for k, v in context.items():
            if k in _SAFE_CONTEXT_KEYS:
                safe[k] = v
            elif k.startswith("step_") and k.endswith("_output"):
                safe[k] = str(v)[:2000]  # enough for the LLM to use as file content

        if "session_history" in context:
            safe["session_history"] = str(context["session_history"])[:300]

        if not safe:
            return ""

        return "\n\nCONTEXT FROM PREVIOUS STEPS:\n" + json.dumps(safe, indent=2, default=str)

    async def _choose_tool(
        self, instruction: str, context: Dict[str, Any]
    ) -> Optional[Dict]:
        avoid_tools     = context.get("avoid_tools", [])
        forced_tool     = context.get("forced_tool")
        preferred_tools = context.get("preferred_tools", [])

        tool_schemas = self._build_tool_schemas(avoid_tools)
        context_str  = self._build_context_str(context)

        preferred_hint = ""
        if preferred_tools:
            preferred_hint = (
                f"Historically successful tools: {', '.join(preferred_tools)}\n"
                "Prefer these when appropriate.\n\n"
            )

        user_prompt = f"{preferred_hint}INSTRUCTION:\n{instruction}{context_str}"

        if forced_tool:
            user_prompt += f"\n\nYOU MUST USE TOOL: {forced_tool}"
        if avoid_tools:
            user_prompt += f"\n\nDO NOT USE THESE TOOLS: {avoid_tools}"

        try:
            result = await call_openai_with_tools(
                system_prompt=_TOOL_SELECTION_SYSTEM,
                user_prompt=user_prompt,
                tools=tool_schemas,
                model=self.model,
                temperature=0.1,
                max_tokens=2000,
            )

            if result["type"] == "tool_call":
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

            # Plain-text fallback
            logger.warning("executor_tool_binding_text_fallback", preview=result.get("content", "")[:200])
            decision = extract_json(result.get("content", ""), context="executor")
            if decision:
                return decision

            logger.error("executor_tool_binding_fallback_failed")
            return None

        except Exception as e:
            logger.error("executor_choice_error", error=str(e))
            return None
