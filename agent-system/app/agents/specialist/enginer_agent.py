import json
import structlog
import time
from typing import Dict, Any, Optional

from app.agents.base_agent import BaseAgent, AgentResult
from app.utils.llm import call_llm_with_system
from app.utils.file_manager import FileManager
from app.utils.json_parser import extract_json
from app.core.config import settings
from app.tools.python_tool import RestrictedPythonExecutor
from app.tools.shell_tool import ShellExecutor
from app.tools.file_tools import (
    FileReadTool, FileWriteTool, FileListTool, FileDeleteTool
)

logger = structlog.get_logger()


class EngineerAgent(BaseAgent):
    """
    Specialist agent for engineering tasks.

    Uses LLM to decide which tool and inputs to use, then
    actually executes the chosen tool (python_executor, file_*, shell_executor).
    """

    SYSTEM_PROMPT = """You are an engineering specialist agent. Your job is to solve technical tasks.

You can:
- Write and execute Python code for calculations, algorithms, data processing
- Create, read, and modify files
- Execute shell commands (limited whitelist)

When given a task, determine the best approach:
1. python_executor — run Python code (calculations, data processing, generating files)
2. file_write — save content to a .txt file
3. file_read — read an existing file
4. file_list — list files in the workspace
5. file_delete — delete a file
6. shell_executor — run safe shell commands (ls, cat, grep, etc.)

RESPONSE FORMAT (JSON only):
{
    "approach": "python_code|file_operation|shell_command",
    "tool": "tool_name",
    "inputs": {
    "param1": "value1"
    },
    "reasoning": "why this approach"
}

For python_executor, provide complete EXECUTABLE Python code in the "code" field.
For file_write, provide exact filename (must end in .txt) and full content.
For shell_executor, provide a valid shell command from the allowed list.

RESPOND ONLY WITH JSON."""

    def __init__(
        self, name: str = "engineer_001", model: str = "claude-haiku-4-5-20251001"
    ):
        super().__init__(
            name=name,
            role="engineer",
            allowed_tools=[
                "python_executor",
                "file_read",
                "file_write",
                "file_list",
                "file_delete",
                "shell_executor",
            ],
        )
        self.model = model

        # Initialise real tools
        fm = FileManager(base_dir=settings.WORKSPACE_DIR)
        self._tools: Dict[str, Any] = {
            "python_executor": RestrictedPythonExecutor(),
            "shell_executor":  ShellExecutor(),
            "file_read":       FileReadTool(fm),
            "file_write":      FileWriteTool(fm),
            "file_list":       FileListTool(fm),
            "file_delete":     FileDeleteTool(fm),
        }

    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        start_time = time.time()
        logger.info("engineer_executing", task=task)

        try:
            tool_decision = await self._decide_approach(task, context)

            if not tool_decision:
                self.record_failure()
                return AgentResult(
                    success=False,
                    output="",
                    errors=["Failed to determine engineering approach"],
                    confidence=0.0,
                    agent_name=self.name,
                    duration_sec=time.time() - start_time,
                )

            approach   = tool_decision.get("approach", "")
            tool_name  = tool_decision.get("tool", "")
            tool_inputs = tool_decision.get("inputs", {})
            reasoning  = tool_decision.get("reasoning", "")

            logger.info(
                "engineer_approach",
                approach=approach,
                tool=tool_name,
                reasoning=reasoning,
            )

            # ── Validate tool is allowed ───────────────────────────────────
            if tool_name not in self._tools:
                self.record_failure()
                return AgentResult(
                    success=False,
                    output="",
                    errors=[f"Unknown or disallowed tool: {tool_name}"],
                    confidence=0.0,
                    agent_name=self.name,
                    duration_sec=time.time() - start_time,
                )

            # ── Execute the tool ───────────────────────────────────────────
            tool = self._tools[tool_name]
            tool_result = await tool.run(**tool_inputs)

            duration = time.time() - start_time

            if tool_result.success:
                logger.info(
                    "engineer_completed",
                    tool=tool_name,
                    output_len=len(tool_result.output),
                    duration=duration,
                )
                self.record_success()
                return AgentResult(
                    success=True,
                    output=tool_result.output,
                    metadata={
                        "approach":  approach,
                        "tool":      tool_name,
                        "reasoning": reasoning,
                        **tool_result.metadata,
                    },
                    confidence=0.85,
                    agent_name=self.name,
                    duration_sec=duration,
                )
            else:
                logger.warning(
                    "engineer_tool_failed",
                    tool=tool_name,
                    error=tool_result.error,
                )
                self.record_failure()
                return AgentResult(
                    success=False,
                    output=tool_result.output or "",
                    errors=[tool_result.error or f"{tool_name} failed"],
                    confidence=0.0,
                    agent_name=self.name,
                    duration_sec=duration,
                )

        except Exception as e:
            logger.error("engineer_error", error=str(e))
            self.record_failure()
            return AgentResult(
                success=False,
                output="",
                errors=[str(e)],
                confidence=0.0,
                agent_name=self.name,
                duration_sec=time.time() - start_time,
            )

    async def _decide_approach(
        self, task: str, context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        context_str = ""
        if context:
            safe_ctx = {
                k: v for k, v in context.items()
                if k in ("task_description", "researcher_output", "week4_output",
                         "should_search", "session_history")
            }
            if safe_ctx:
                context_str = f"\n\nCONTEXT:\n{json.dumps(safe_ctx, indent=2, default=str)[:1000]}"

        user_prompt = (
            f"ENGINEERING TASK:\n{task}{context_str}\n\n"
            "Determine the best approach and provide exact, executable inputs.\n"
            "Return JSON only."
        )

        try:
            response = await call_llm_with_system(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self.model,
                temperature=0.1,
            )

            decision = extract_json(response, context="engineer")
            if not decision:
                logger.error("engineer_decision_json_error",
                             response_preview=(response or "")[:200])
                return None

            logger.debug(
                "engineer_decision",
                approach=decision.get("approach"),
                tool=decision.get("tool"),
            )
            return decision

        except Exception as e:
            logger.error("engineer_decision_error", error=str(e))
            return None
