import subprocess
import os
import structlog
from typing import Dict, Any, List

from app.tools.base import Tool, ToolResult

logger = structlog.get_logger()

class ShellExecutor(Tool):
    """
    Execute safe shell commands.
    
    SECURITY:
    - Whitelist of allowed commands
    - Timeout protection
    - Working directory isolation
    """
    
    # Whitelist of safe commands
    ALLOWED_COMMANDS = {
        "ls", "pwd", "cat", "grep", "find", "wc", "head", "tail",
        "echo", "mkdir", "touch", "cp", "mv", "tree", "du", "df"
    }
    
    @property
    def name(self) -> str:
        return "shell_executor"
    
    @property
    def description(self) -> str:
        return f"Execute safe shell commands. Allowed: {', '.join(self.ALLOWED_COMMANDS)}"
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        }
    
    def _is_command_safe(self, command: str) -> bool:
        """Check if command is in whitelist"""
        parts = command.strip().split()
        if not parts:
            return False
        
        base_command = parts[0]
        return base_command in self.ALLOWED_COMMANDS
    
    async def run(self, **kwargs) -> ToolResult:
        """Execute shell command safely"""
        command = kwargs.get("command", "")
        
        if not command.strip():
            return ToolResult(
                success=False,
                output="",
                error="No command provided"
            )
         
        # Security check
        if not self._is_command_safe(command):
            base_cmd = command.split()[0] if command.split() else ""
            return ToolResult(
                success=False,
                output="",
                error=f"Command '{base_cmd}' not allowed. Allowed: {', '.join(self.ALLOWED_COMMANDS)}"
            )
        
        logger.info("shell_executor_running", command=command)
        
        sandbox_dir = os.getenv("SANDBOX_DIR", "/app/sandbox")
        os.makedirs(sandbox_dir, exist_ok=True)
        
        try:
            shared_workspace = os.getenv("SHARED_WORKSPACE", "/app/workspace/shared"),
            os.makedirs(shared_workspace, exist_ok=True)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=shared_workspace  # Isolate in shared workspace
                )
            
            if result.returncode == 0:
                logger.info("shell_executor_success", output_length=len(result.stdout))
                return ToolResult(
                    success=True,
                    output=result.stdout,
                    metadata={"return_code": 0}
                )
            else:
                logger.warning("shell_executor_failed", error=result.stderr)
                return ToolResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    metadata={"return_code": result.returncode}
                )
        
        except subprocess.TimeoutExpired:
            logger.error("shell_executor_timeout")
            return ToolResult(
                success=False,
                output="",
                error="Command timed out after 30 seconds"
            )
        
        except Exception as e:
            logger.error("shell_executor_error", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}"
            )