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
        "ls", "pwd", "cat", "grep", "wc", "head", "tail",
        "echo", "mkdir", "touch", "tree", "du", "df",
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
        try:
            parts = shlex.split(command)
        except ValueError:
            return False
        if not parts or parts[0] not in self.ALLOWED_COMMANDS:
            return False
        # Keep every relative path inside the workspace. Remove commands with
        # recursive/executable behavior (find/cp/mv) from the allowlist above.
        for argument in parts[1:]:
            path_parts = argument.replace("\\", "/").split("/")
            if ".." in path_parts or argument.startswith(("/", "\\\\")):
                return False
            if (len(argument) >= 2 and argument[1] == ":") or argument.startswith("~"):
                return False
        return True

    async def run(self, **kwargs: Any) -> ToolResult:
        command: str = kwargs.get("command", "")

        if not command.strip():
            return ToolResult(success=False, output="", error="No command provided")

        if not self._is_command_safe(command):
            try:
                parsed_command = shlex.split(command)
            except ValueError:
                parsed_command = []
            base_cmd = parsed_command[0] if parsed_command else ""
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Command '{base_cmd}' not allowed. "
                    f"Allowed: {', '.join(sorted(self.ALLOWED_COMMANDS))}"
                ),
            )

        logger.info("shell_executor_running", command_name=shlex.split(command)[0])

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
                logger.warning("shell_executor_failed", return_code=result.returncode)
                return ToolResult(
                    success=False,
                    output=result.stdout,
                    error=f"Command failed with exit code {result.returncode}",
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
            logger.error("shell_executor_error", error_type=type(e).__name__)
            return ToolResult(
                success=False,
                output="",
                error=f"Execution error ({type(e).__name__})",
            )
