import json
import structlog
from typing import Dict, Any

from app.utils.llm import call_llm
from app.tools.base import Tool, ToolResult
from app.tools.python_tool import PythonExecutor
from app.tools.shell_tool import ShellExecutor

logger = structlog.get_logger()


class ExecutorAgent:
    """
    Chooses and executes tools to complete steps.

    RESPONSIBILITIES:
    1. Understand step instruction
    2. Select appropriate tool
    3. Generate tool inputs
    4. Execute tool
    5. Return structured result
    """

    SYSTEM_PROMPT = """You are a precise tool execution agent. Your job is to:
1. Read the step instruction
2. Choose the RIGHT tool
3. Generate the EXACT EXECUTABLE inputs needed

CRITICAL: For python_executor, you MUST provide actual executable Python code, NOT the instruction text!

AVAILABLE TOOLS:
{tools_description}

RESPONSE FORMAT (JSON only):
{{
  "tool": "tool_name",
  "inputs": {{
    "param1": "value1"
  }},
  "reasoning": "why this tool and these inputs"
}}

EXAMPLES:

INSTRUCTION: "Create a Python script that calculates the sum of numbers from 1 to 100"
CORRECT RESPONSE:
{{
  "tool": "python_executor",
  "inputs": {{
    "code": "total = sum(range(1, 101))\\nprint(f'Sum: {{total}}')"
  }},
  "reasoning": "Using Python's built-in sum function to calculate the sum efficiently"
}}

WRONG RESPONSE (DO NOT DO THIS):
{{
  "tool": "python_executor",
  "inputs": {{
    "code": "Create a Python script that calculates the sum of numbers from 1 to 100"
  }},
  "reasoning": "..."
}}

INSTRUCTION: "List all Python files in the current directory"
CORRECT RESPONSE:
{{
  "tool": "shell_executor",
  "inputs": {{
    "command": "ls *.py"
  }},
  "reasoning": "Using ls with wildcard to list Python files"
}}

RULES:
- For python_executor: "code" must be EXECUTABLE Python code with proper syntax
- For shell_executor: "command" must be a valid shell command
- No placeholders or TODO comments
- Include print() statements to show results
- Use \\n for line breaks in code strings

RESPOND ONLY WITH JSON.
"""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        # Claude is excellent for tool reasoning
        self.model = model
        self.tools: Dict[str, Tool] = {}

        self._register_tool(PythonExecutor())
        self._register_tool(ShellExecutor())

    def _register_tool(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.info("tool_registered", tool=tool.name)

    def _get_tools_description(self) -> str:
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(
                f"""
Tool: {tool.name}
Description: {tool.description}
Input Schema: {json.dumps(tool.input_schema, indent=2)}

IMPORTANT for {tool.name}:
{"- The 'code' parameter must be EXECUTABLE Python code, not a description" if tool.name == "python_executor" else ""}
{"- The 'command' parameter must be a valid shell command, not a description" if tool.name == "shell_executor" else ""}
"""
            )
        return "\n".join(descriptions)

    async def execute_step(
        self, instruction: str, context: Dict[str, Any] | None = None
    ) -> ToolResult:

        logger.info("executor_starting", instruction=instruction)
        context = context or {}

        # ----------------------------------
        # 🔥 DETERMINISTIC FALLBACK (NO LLM)
        # ----------------------------------
        instruction_l = instruction.lower()

        try:
            # Simple heuristics for common cases
            if instruction_l.startswith("list") or "list files" in instruction_l:
                return await self.tools["shell_executor"].run(command="ls -la")

        except Exception as e:
            logger.warning(
                "executor_fallback_failed",
                instruction=instruction,
                error=str(e),
            )

        # ----------------------------------
        # LLM TOOL SELECTION (Claude)
        # ----------------------------------
        tool_decision = await self._choose_tool(instruction, context)

        if not tool_decision:
            return ToolResult(
                success=False,
                output="",
                error="Failed to choose appropriate tool",
            )

        tool_name = tool_decision.get("tool")
        tool_inputs = tool_decision.get("inputs", {})
        reasoning = tool_decision.get("reasoning", "")

        logger.info(
            "executor_tool_selected",
            tool=tool_name,
            reasoning=reasoning,
        )

        # Validate tool exists
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        # Additional validation for python_executor
        if tool_name == "python_executor":
            code = tool_inputs.get("code", "")
            if not code:
                return ToolResult(
                    success=False,
                    output="",
                    error="Python executor requires 'code' parameter",
                )
            # Check if code looks like an instruction rather than code
            if code.lower().startswith(("create a", "write a", "make a", "build a")):
                logger.warning(
                    "executor_invalid_code_input",
                    code_preview=code[:100],
                )
                return ToolResult(
                    success=False,
                    output="",
                    error="Received instruction text instead of executable code. Please provide actual Python code.",
                )

        tool = self.tools[tool_name]

        try:
            result = await tool.run(**tool_inputs)
            logger.info(
                "executor_completed",
                tool=tool_name,
                success=result.success,
            )
            return result

        except Exception as e:
            logger.error("executor_error", tool=tool_name, error=str(e))
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution failed: {str(e)}",
            )

    async def _choose_tool(
        self, instruction: str, context: Dict[str, Any]
    ) -> Dict | None:

        tools_desc = self._get_tools_description()
        system_prompt = self.SYSTEM_PROMPT.format(
            tools_description=tools_desc
        )

        context_str = ""
        if context:
            context_str = (
                "\n\nCONTEXT FROM PREVIOUS STEPS:\n"
                + json.dumps(context, indent=2)
            )

        user_prompt = f"""STEP INSTRUCTION:
{instruction}
{context_str}

Choose the appropriate tool and generate EXECUTABLE inputs (not instruction text).
For python_executor, generate actual Python code.
For shell_executor, generate actual shell commands.
Return JSON only.
"""

        try:
            response = await call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model,
                temperature=0.1,
            )

            response_text = response.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1] if "\n" in response_text else response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
            
            start = response_text.find("{")
            end = response_text.rfind("}")

            if start == -1 or end == -1 or end <= start:
                raise json.JSONDecodeError(
                    "No valid JSON object found", response_text, 0
                )

            decision = json.loads(response_text[start : end + 1])
            
            # Log what we're about to execute for debugging
            if "inputs" in decision:
                logger.debug(
                    "executor_tool_decision",
                    tool=decision.get("tool"),
                    inputs_preview=str(decision["inputs"])[:200],
                )

            return decision

        except json.JSONDecodeError as e:
            logger.error("executor_json_error", error=str(e), response=response_text[:500])
            return None

        except Exception as e:
            logger.error("executor_choice_error", error=str(e))
            return None