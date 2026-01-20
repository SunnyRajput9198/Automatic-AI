import json
import structlog
from typing import Dict, Any

from app.utils.llm import call_llm
from app.utils.file_manager import FileManager
from app.tools.base import Tool, ToolResult
from app.tools.python_tool import PythonExecutor
from app.tools.shell_tool import ShellExecutor
from app.tools.file_tools import FileReadTool, FileWriteTool, FileListTool, FileDeleteTool
from app.tools.web_search import WebSearchTool, WebFetchTool

logger = structlog.get_logger()


class ExecutorAgent:
    """
    Enhanced executor with Week 2 tools + your custom improvements.
    
    WEEK 1 TOOLS:
    - python_executor
    - shell_executor
    
    WEEK 2 TOOLS:
    - file_read, file_write, file_list, file_delete (persistent workspace)
    - web_search, web_fetch (web access)
    
    YOUR ENHANCEMENTS:
    - Claude Haiku 4.5 model
    - Deterministic fallback for common cases
    - Enhanced validation for python_executor
    """

    SYSTEM_PROMPT = """You are a precise tool execution agent. Your job is to:
    1. Read the step instruction
    2. Choose the RIGHT tool
    3. Generate the EXACT EXECUTABLE inputs needed

    CRITICAL: For python_executor, you MUST provide actual executable Python code, NOT the instruction text!

    AVAILABLE TOOLS:
    {tools_description}

    IMPORTANT NOTES ABOUT web_search:
    - web_search ALREADY returns formatted, structured results
    - Output includes: numbered results, titles, descriptions, and URLs
    - NO parsing or extraction needed - results are ready to use
    - If you need to analyze results, read them directly - don't try to parse them

    RESPONSE FORMAT (JSON only):
    {{
    "tool": "tool_name",
    "inputs": {{
        "param1": "value1"
    }},
    "reasoning": "why this tool and these inputs"
    }}
 IMPORTANT:
If a task requires external libraries (flask, fastapi, django, etc),
DO NOT execute them.
Instead, generate code as a file or plain output without running it.
    For code in JSON strings:
    - Use escaped newlines: \\n
    - Keep code simple and focused
    - Avoid embedding large data structures in code

    RESPOND ONLY WITH VALID JSON.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        # Your choice: Claude Haiku for excellent tool reasoning
        self.model = model
        self.tools: Dict[str, Tool] = {}
        
        # Initialize file manager (Week 2)
        self.file_manager = FileManager()

        # Register Week 1 tools
        self._register_tool(PythonExecutor())
        self._register_tool(ShellExecutor())
        
        # Register Week 2 file tools
        self._register_tool(FileReadTool(self.file_manager))
        self._register_tool(FileWriteTool(self.file_manager))
        self._register_tool(FileListTool(self.file_manager))
        self._register_tool(FileDeleteTool(self.file_manager))
        
        # Register Week 2 web tools
        self._register_tool(WebSearchTool())
        self._register_tool(WebFetchTool())

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
{"- Files persist across tasks in shared workspace" if tool.name.startswith("file_") else ""}
"""
            )
        return "\n".join(descriptions)

    async def execute_step(
        self, instruction: str, context: Dict[str, Any] | None = None
    ) -> ToolResult:

        logger.info("executor_starting", instruction=instruction)
        context = context or {}

        # ----------------------------------
        # 🔥 YOUR DETERMINISTIC FALLBACK (NO LLM)
        # ----------------------------------
        instruction_l = instruction.lower()

        try:
            # Simple heuristics for common cases
            if instruction_l.startswith("list") or "list files" in instruction_l:
                # Week 2: Check if they want workspace files or sandbox files
                if "workspace" in instruction_l or "persistent" in instruction_l:
                    return await self.tools["file_list"].run()
                else:
                    return await self.tools["shell_executor"].run(command="ls -la")

        except Exception as e:
            logger.warning(
                "executor_fallback_failed",
                instruction=instruction,
                error=str(e),
            )

        # ----------------------------------
        # LLM TOOL SELECTION (Your Claude Model)
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

        # YOUR VALIDATION: Additional check for python_executor
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
    For file_* tools, use appropriate filenames and content.
    For web_* tools, use proper queries or URLs.
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
                max_tokens=4000,  # Increase token limit for longer responses
            )

            response_text = response.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()
            
            # Try to find and extract valid JSON
            # Method 1: Direct parse
            try:
                decision = json.loads(response_text)
                logger.debug(
                    "executor_tool_decision",
                    tool=decision.get("tool"),
                    inputs_preview={k: str(v)[:100] for k, v in decision.get("inputs", {}).items()}
                )
                return decision
            except json.JSONDecodeError:
                pass
            
            # Method 2: Find JSON object boundaries
            import re
            
            # Find the outermost JSON object
            brace_count = 0
            start_idx = -1
            end_idx = -1
            
            for i, char in enumerate(response_text):
                if char == '{':
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        end_idx = i
                        break
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx + 1]
                try:
                    decision = json.loads(json_str)
                    logger.debug(
                        "executor_tool_decision",
                        tool=decision.get("tool"),
                        inputs_preview={k: str(v)[:100] for k, v in decision.get("inputs", {}).items()}
                    )
                    return decision
                except json.JSONDecodeError as e:
                    logger.error(
                        "executor_json_parse_error",
                        error=str(e),
                        json_preview=json_str[:500],
                        position=e.pos if hasattr(e, 'pos') else None
                    )
            
            # If all parsing failed, log the full response for debugging
            logger.error(
                "executor_json_extraction_failed",
                response_length=len(response_text),
                response_preview=response_text[:1000],
                response_end=response_text[-500:] if len(response_text) > 500 else response_text
            )
            return None

        except Exception as e:
            logger.error("executor_choice_error", error=str(e))
            return None