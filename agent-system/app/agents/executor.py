import re
import json
import time
import structlog
from typing import Dict, Any, Optional

from app.utils.json_parser import extract_json
from app.utils.llm import call_openai_with_system, call_openai_with_tools
from app.utils.file_manager import FileManager
from app.utils.cost_tracker import global_cost_tracker
from app.tools.base import Tool, ToolResult, classify_tool_failure
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
from app.tools.github_tool import GitHubTool
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
    "python_executor", "restricted_python_executor", "shell_executor", "github",
})

_VALID_TOOL_NAMES = frozenset({
    "web_search", "web_fetch", "github", "get_weather", "news_search",
    "semantic_scholar_search", "wikipedia_search", "python_executor",
    "shell_executor", "file_read", "file_write", "file_append", "file_list", "file_delete",
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
    "- github                  → latest commit or release from a named repository; do not use web_search for this\n"
    "- get_weather             → current temperature and weather for any city — ALWAYS use this for temperature/weather queries\n"
    "- web_search              → ambiguous or broad queries needing multiple sources\n"
    "- web_fetch               → fetch content from a specific known URL\n"
    "- news_search             → current news, today's headlines, breaking news, recent events\n"
    "- python_executor         → run executable Python code (provide actual code, not a description)\n"
    "- Use only the minimum necessary tool. A specific GitHub request maps to github; weather maps to get_weather; fresh general information maps to web_search.\n"
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
            GitHubTool(),
        ]:
            self._register(tool)

    def _register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        logger.info("tool_registered", tool=tool.name)

    @staticmethod
    def can_fallback(primary: str, fallback: str, request: str) -> bool:
        """Allow only source alternatives that can answer the same information need."""
        compatible = {
            "github": {"web_search"},
            "semantic_scholar_search": {"web_search"},
            "wikipedia_search": {"web_search"},
            "web_search": {"news_search", "semantic_scholar_search"},
        }
        if fallback not in compatible.get(primary, set()):
            return False
        # Search indexes cannot reliably establish private repo state. Only
        # permit an API-to-web fallback if the request identifies public data.
        if primary == "github" and (
            "public" not in request.lower() or "private" in request.lower()
        ):
            return False
        return True

    async def execute_with_fallbacks(
        self,
        instruction: str,
        primary_tool: str,
        fallback_tools: list,
        request: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Run the selected wrapper, then only explicitly compatible alternatives."""
        context = dict(context or {})
        context["forced_tool"] = primary_tool
        primary_result = await self.execute_step(instruction, context=context)
        attempts = [(primary_tool, primary_result)]
        candidates = list(fallback_tools or [])
        max_attempts = max(1, int(context.get("max_tool_attempts", 1 + len(candidates))))
        if primary_tool == "github" and "public" in request.lower():
            candidates.append("web_search")
        if primary_result.success:
            return primary_result, attempts

        final_result = primary_result
        for fallback_name in dict.fromkeys(candidates):
            if len(attempts) >= max_attempts:
                break
            if not self.can_fallback(primary_tool, fallback_name, request):
                continue
            if fallback_name not in self.tools:
                continue
            fallback_instruction = (
                "Search official GitHub web pages for the latest commit or release of the explicitly "
                "public repository in this request. Verify the SHA or tag from a direct GitHub URL; "
                "if it cannot be verified, return no result.\n"
                if primary_tool == "github" else
                f"Use {fallback_name} as an alternative source for the same information. "
                "Return only evidence this source actually provides.\n"
            ) + f"User request: {request}"
            fallback_context = dict(context)
            fallback_context["forced_tool"] = fallback_name
            fallback_result = await self.execute_step(fallback_instruction, context=fallback_context)
            attempts.append((fallback_name, fallback_result))
            if fallback_result.success:
                if primary_tool == "github" and not self._github_search_evidence_matches(request, fallback_result.output):
                    fallback_result.metadata["validation_status"] = "unverified"
                    fallback_result.metadata["failure_type"] = "UNVERIFIED_FALLBACK"
                    fallback_result.error = "Web search did not verify the requested repository commit or release"
                    final_result = ToolResult(
                        success=False, output="", error=fallback_result.error,
                        metadata={"tool_name": primary_tool, "source": "GitHub REST API",
                                  "failure_type": "UNVERIFIED_FALLBACK"},
                    )
                    continue
                fallback_result.metadata["fallback_from"] = primary_tool
                return fallback_result, attempts
            final_result = fallback_result
        return final_result, attempts

    @staticmethod
    def _github_search_evidence_matches(request: str, output: str) -> bool:
        """Require repo-specific GitHub links and an identifiable commit or release tag."""
        match = re.search(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", request)
        if not match:
            return False
        repository = f"{match.group(1)}/{match.group(2)}".lower()
        text = output.lower()
        if repository not in text or f"github.com/{repository}" not in text:
            return False
        if "release" in request.lower():
            return "/releases/" in text and ("tag" in text or "release" in text)
        return bool(re.search(r"(?<![a-f0-9])[a-f0-9]{7,40}(?![a-f0-9])", text))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def execute_step(
        self, instruction: str, context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """Choose a tool via LLM tool binding and execute a single plan step."""
        logger.info("executor_starting", instruction_length=len(instruction))
        context       = context or {}
        avoid_tools   = context.get("avoid_tools", [])
        instruction_l = instruction.lower()

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
        if tool_name in {"python_executor", "restricted_python_executor"}:
            code = tool_inputs.get("code", "")
            if not code:
                return ToolResult(
                    success=False, output="",
                    error=f"{tool_name} requires a 'code' parameter.",
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
            result.metadata.setdefault("tool_name", tool_name)
            result.metadata.setdefault(
                "source", result.metadata.get("sources") or tool_name
            )
            safe_query = next((tool_inputs[key] for key in ("query", "city", "owner") if tool_inputs.get(key)), None)
            if safe_query:
                result.metadata.setdefault("query", str(safe_query)[:500])
            if result.success:
                from datetime import datetime, timezone
                result.metadata.setdefault("retrieved_at", datetime.now(timezone.utc).isoformat())
            else:
                result.metadata.setdefault(
                    "failure_type", classify_tool_failure(result.error)
                )
            global_cost_tracker.record_tool_call(
                tool_name=tool_name,
                agent="executor",
                success=result.success,
                duration_ms=(time.time() - t0) * 1000,
            )
            logger.info("executor_completed", tool=tool_name, success=result.success)
            return result
        except Exception as e:
            failure_type = classify_tool_failure(type(e).__name__)
            logger.error("executor_error", tool=tool_name, failure_type=failure_type)
            return ToolResult(success=False, output="", error=f"Tool execution failed ({failure_type})",
                              metadata={"tool_name": tool_name, "source": tool_name, "failure_type": failure_type})

    # ------------------------------------------------------------------
    # Private: LLM tool binding
    # ------------------------------------------------------------------

    def _build_tool_schemas(self, avoid_tools: list, only_tool: Optional[str] = None) -> list:
        return [
            tool.to_openai_schema()
            for name, tool in self.tools.items()
            if name not in avoid_tools and (only_tool is None or name == only_tool)
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

        tool_schemas = self._build_tool_schemas(avoid_tools, only_tool=forced_tool)
        if forced_tool and not tool_schemas:
            logger.warning("planner_selected_unavailable_tool", tool=forced_tool)
            return None
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
            logger.error("executor_choice_error", error_type=type(e).__name__)
            return None
