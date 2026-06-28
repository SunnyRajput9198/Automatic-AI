import subprocess
import os
import structlog
import shlex
from typing import Any, Dict

from app.core.config import settings
from app.tools.base import Tool, ToolResult

logger = structlog.get_logger()


class ShellExecutor(Tool):
    """
    Execute a whitelisted subset of shell commands in the shared workspace.

    Security model:
    - Only commands in ALLOWED_COMMANDS can run (checked on the base word).
    - Command chaining characters are blocked to prevent injection.
    - Arguments are passed as a list with shell=False — the OS never
      interprets the command string, eliminating shell injection risk.
    - Execution is capped at 30 seconds.
    - Working directory is set via the native cwd= param so file operations
      land in the shared workspace.
    """

    ALLOWED_COMMANDS = {
        "ls", "pwd", "cat", "grep", "find", "wc", "head", "tail",
        "echo", "mkdir", "touch", "cp", "mv", "tree", "du", "df",
    }

    # Characters that enable command chaining / injection
    _DANGEROUS = [";", "&&", "||", "|", "`", "$("]

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
        """Return False if command contains chaining/injection characters."""
        if any(char in command for char in self._DANGEROUS):
            return False
        parts = shlex.split(command)
        return bool(parts) and parts[0] in self.ALLOWED_COMMANDS

    async def run(self, **kwargs: Any) -> ToolResult:
        command: str = kwargs.get("command", "")

        if not command.strip():
            return ToolResult(success=False, output="", error="No command provided")

        if not self._is_command_safe(command):
            base_cmd = shlex.split(command)[0] if shlex.split(command) else ""
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Command '{base_cmd}' not allowed. "
                    f"Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS))}"
                ),
            )

        logger.info("shell_executor_running", command=command)

        shared_workspace: str = settings.SHARED_WORKSPACE
        os.makedirs(shared_workspace, exist_ok=True)

        try:
            result = subprocess.run(
                shlex.split(command),   # list of args — shell=False is safe
                shell=False,            # OS never interprets the string
                capture_output=True,
                text=True,
                timeout=30,             # bumped from 10s — find/du can be slow
                cwd=shared_workspace,   # guaranteed cwd, no cd-chain workaround
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
            logger.error("shell_executor_timeout", command=command)
            return ToolResult(
                success=False,
                output="",
                error="Command timed out after 30 seconds",
            )

        except Exception as e:
            logger.error("shell_executor_error", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
            )
