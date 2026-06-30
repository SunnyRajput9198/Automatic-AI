from RestrictedPython import compile_restricted, safe_builtins, utility_builtins
from RestrictedPython.PrintCollector import PrintCollector
from app.tools.base import Tool, ToolResult
import structlog
from typing import Dict, Any

logger = structlog.get_logger()

class RestrictedPythonExecutor(Tool):
    """
    Execute Python code safely using RestrictedPython.

    SECURITY:
    - Blocks dangerous builtins (file I/O, subprocess, etc.)
    - Provides controlled globals via RestrictedPython's safe_builtins
    - print() output is captured via PrintCollector
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
                "_print_": PrintCollector,  # capture print output
                "_getattr_": getattr,
                "_setattr_": setattr,
                "_getitem_": lambda obj, key: obj[key],
                "_setitem_": lambda obj, key, value: obj.__setitem__(key, value),
            }

            # Prepare locals with a print collector
            safe_locals = {}
            exec(byte_code, safe_globals, safe_locals)

            # Collect printed output
            output = safe_locals["_print_"].read()

            logger.info("restricted_python_executor_success", output_length=len(output))
            return ToolResult(success=True, output=output, metadata={"sandbox": "RestrictedPython"})

        except Exception as e:
            logger.error("restricted_python_executor_error", error=str(e))
            return ToolResult(success=False, output="", error=f"Execution error: {str(e)}")
