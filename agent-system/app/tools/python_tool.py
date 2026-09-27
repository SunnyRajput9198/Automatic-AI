from RestrictedPython import compile_restricted, safe_builtins, utility_builtins
from RestrictedPython.PrintCollector import PrintCollector
from app.tools.base import Tool, ToolResult
import importlib
import structlog
from typing import Dict, Any

logger = structlog.get_logger()
_SAFE_MODULES = frozenset({"math", "statistics"})


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Allow only pure computation modules inside the sandbox."""
    if level or name not in _SAFE_MODULES:
        raise ImportError("Only math and statistics imports are available")
    return importlib.import_module(name)

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
        return "python_executor"

    @property
    def description(self) -> str:
        return "Run actual Python calculations or data processing in an isolated RestrictedPython sandbox; returns captured stdout. Use for meaningful computation, not trivial arithmetic. Required input: executable code string. It cannot perform normal host file/network access."

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
                "__builtins__": safe_builtins | utility_builtins | {"__import__": _restricted_import},
                "_print_": PrintCollector,  # capture print output
                "_getattr_": getattr,
                "_setattr_": setattr,
                "_getitem_": lambda obj, key: obj[key],
                "_setitem_": lambda obj, key, value: obj.__setitem__(key, value),
            }

            # Prepare locals with a print collector
            safe_locals = {}
            exec(byte_code, safe_globals, safe_locals)

            # RestrictedPython stores the print collector in `_print` within
            # locals after executing generated print-hook calls.
            print_collector = safe_locals.get("_print")
            output = print_collector() if print_collector else ""

            logger.info("restricted_python_executor_success", output_length=len(output))
            return ToolResult(success=True, output=output, metadata={"sandbox": "RestrictedPython"})

        except Exception as e:
            logger.error("restricted_python_executor_error", error=str(e))
            return ToolResult(success=False, output="", error=f"Execution error: {str(e)}")
