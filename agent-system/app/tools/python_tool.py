import subprocess
import os
import tempfile
import structlog
from typing import Dict, Any

from app.tools.base import Tool, ToolResult

logger = structlog.get_logger()

class PythonExecutor(Tool):
    """
    Execute Python code in a sandboxed environment.
    
    SECURITY:
    - Runs in isolated subprocess
    - Timeout protection
    - Working directory isolation
    """
    
    @property
    def name(self) -> str:
        return "python_executor"
    
    @property
    def description(self) -> str:
        return "Execute Python code in a sandbox. Use for data processing, calculations, file reading, etc."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    
    async def run(self, **kwargs) -> ToolResult:
        """Execute Python code safely"""
        code = kwargs.get("code", "")
        
        if not code.strip():
            return ToolResult(
                success=False,
                output="",
                error="No code provided"
            )
        
        logger.info("python_executor_running", code_length=len(code))
        
        # Create sandbox directory
        sandbox_dir = os.getenv("SANDBOX_DIR", "/app/sandbox")
        os.makedirs(sandbox_dir, exist_ok=True)
        
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                dir=sandbox_dir,
                delete=False
            ) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute with timeout
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=sandbox_dir
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                logger.info("python_executor_success", output_length=len(result.stdout))
                return ToolResult(
                    success=True,
                    output=result.stdout,
                    metadata={"return_code": 0}
                )
            else:
                logger.warning("python_executor_failed", error=result.stderr)
                return ToolResult(
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    metadata={"return_code": result.returncode}
                )
        
        except subprocess.TimeoutExpired:
            logger.error("python_executor_timeout")
            return ToolResult(
                success=False,
                output="",
                error="Execution timed out after 30 seconds"
            )
        
        except Exception as e:
            logger.error("python_executor_error", error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Execution error: {str(e)}"
            )