import subprocess
import os
import structlog
from typing import Any, Dict

from app.tools.base import Tool, ToolResult

logger = structlog.get_logger()
from app.core.config import settings


class ShellExecutor(Tool):
    """
    Execute a whitelisted subset of shell commands in the shared workspace.

    Security model:
    - Only commands in ALLOWED_COMMANDS can run (checked on the base word).
    - Execution is capped at 30 seconds.
    - Working directory is set to the shared workspace so file operations
    land in the right place.
    """

    ALLOWED_COMMANDS = {
        "ls", "pwd", "cat", "grep", "find", "wc", "head", "tail",
        "echo", "mkdir", "touch", "cp", "mv", "tree", "du", "df",
    }

    @property
    def name(self) -> str:
        return "shell_executor"

    @property
    def description(self) -> str:
        return (
            "Execute safe shell commands. "
            f"Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS))}"
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                }
            },
            "required": ["command"],
        }

    def _is_command_safe(self, command: str) -> bool:
        # block command chaining characters
        dangerous_chars = [";", "&&", "||", "|", "`", "$(" ]
        for char in dangerous_chars:
            if char in command:
                return False
        parts = command.strip().split()
        return bool(parts) and parts[0] in self.ALLOWED_COMMANDS

    async def run(self, **kwargs: Any) -> ToolResult:
        command: str = kwargs.get("command", "")

        if not command.strip():
            return ToolResult(success=False, output="", error="No command provided")

        if not self._is_command_safe(command):
            base_cmd = command.split()[0] if command.split() else ""
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Command '{base_cmd}' not allowed. "
                    f"Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS))}"
                ),
            )

        logger.info("shell_executor_running", command=command)

        # FIX: removed trailing comma that made shared_workspace a tuple.
        # Was:  shared_workspace = os.getenv(...),   →  type: tuple[str]
        # Now:  shared_workspace = os.getenv(...)    →  type: str
        # The tuple caused subprocess.run(cwd=...) and os.makedirs(...) to
        # receive a tuple instead of a str, triggering "No overloads match".
        shared_workspace: str = settings.SHARED_WORKSPACE
        os.makedirs(shared_workspace, exist_ok=True)

        try:
            result = subprocess.run(
                command.split(),# split the command into a list of arguments
                shell=False,# tell the subprocess to treat the command as a list of arguments
                capture_output=True,
                text=True,
                timeout=10,
                cwd=shared_workspace,
            )

            if result.returncode == 0:
                logger.info("shell_executor_success", output_length=len(result.stdout))
                return ToolResult(
                    success=True,
                    output=result.stdout,
                    metadata={"return_code": 0},
                )
            else:
                logger.warning("shell_executor_failed", error=result.stderr)
                return ToolResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    metadata={"return_code": result.returncode},
                )

        except subprocess.TimeoutExpired:
            logger.error("shell_executor_timeout")
            return ToolResult(
                success=False,
                output="",
                error="Command timed out after 10 seconds",
            )

        except Exception as e:
            logger.error("shell_executor_error", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
            )