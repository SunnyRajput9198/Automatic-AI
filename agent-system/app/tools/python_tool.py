from RestrictedPython import compile_restricted, safe_builtins, utility_builtins
from app.tools.base import Tool, ToolResult
import structlog
from typing import Dict, Any

logger = structlog.get_logger()

class RestrictedPythonExecutor(Tool):
    """
    Execute Python code safely using RestrictedPython.

    SECURITY:
    - Blocks dangerous builtins (file I/O, subprocess, etc.)
    - Provides controlled globals
    - Faster than subprocess since it runs inline
    """

    @property
    def name(self) -> str:
        return "restricted_python_executor"

    @property
    def description(self) -> str:
        return "Execute Python code safely using RestrictedPython sandbox."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        }

    async def run(self, **kwargs) -> ToolResult:
        code = kwargs.get("code", "")
        if not code.strip():
            return ToolResult(success=False, output="", error="No code provided")

        logger.info("restricted_python_executor_running", code_length=len(code))

        try:
            # Compile code in restricted mode
            byte_code = compile_restricted(code, filename="<inline>", mode="exec")

            # Safe globals
            safe_globals = {
                "__builtins__": safe_builtins | utility_builtins,
                "_print_": lambda *args: " ".join(map(str, args)),  # capture print
                "_getattr_": getattr,
                "_setattr_": setattr,
                "_getitem_": lambda obj, key: obj[key],
                "_setitem_": lambda obj, key, value: obj.__setitem__(key, value),
            }

            # Capture output
            output_buffer = []
            safe_globals["_print_"] = lambda *args: output_buffer.append(" ".join(map(str, args)))

            exec(byte_code, safe_globals)

            result_output = "\n".join(output_buffer)
            logger.info("restricted_python_executor_success", output_length=len(result_output))

            return ToolResult(success=True, output=result_output, metadata={"sandbox": "RestrictedPython"})

        except Exception as e:
            logger.error("restricted_python_executor_error", error=str(e))
            return ToolResult(success=False, output="", error=f"Execution error: {str(e)}")
