import json
import structlog
import time
from typing import Dict, Any, Optional

from app.agents.base_agent import BaseAgent, AgentResult
from app.utils.llm import call_llm

logger = structlog.get_logger()


class EngineerAgent(BaseAgent):
    """
    Specialist agent for engineering tasks.
    
    Generates and executes code, performs calculations,
    manages files using LLM-powered tool selection.
    """
    
    SYSTEM_PROMPT = """You are an engineering specialist agent. Your job is to solve technical tasks.

You can:
- Write Python code for calculations, algorithms, data processing
- Create, read, and modify files
- Execute shell commands (limited)

When given a task, determine the best approach:
1. Python code execution (python_executor)
2. File operations (file_read, file_write, file_list)
3. Shell commands (shell_executor) - use sparingly

RESPONSE FORMAT (JSON only):
{
  "approach": "python_code|file_operation|shell_command",
  "tool": "tool_name",
  "inputs": {
    "param1": "value1"
  },
  "reasoning": "why this approach"
}

For python_executor, provide EXECUTABLE code, not descriptions.
For file operations, provide exact filenames and content.

RESPOND ONLY WITH JSON."""
    
    def __init__(
        self,
        name: str = "engineer_001",
        model: str = "claude-haiku-4-5-20251001"
    ):
        super().__init__(
            name=name,
            role="engineer",
            allowed_tools=[
                "python_executor",
                "file_read",
                "file_write",
                "file_list",
                "file_delete",
                "shell_executor"
            ]
        )
        self.model = model
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute engineering task.
        
        Args:
            task: Engineering task description
            context: Optional context from coordinator
            
        Returns:
            AgentResult with execution output
        """
        start_time = time.time()
        
        logger.info("engineer_executing", task=task)
        
        try:
            # Use LLM to decide approach and generate inputs
            tool_decision = await self._decide_approach(task, context)
            
            if not tool_decision:
                self.record_failure()
                return AgentResult(
                    success=False,
                    output="",
                    errors=["Failed to determine engineering approach"],
                    confidence=0.0,
                    agent_name=self.name,
                    duration_sec=time.time() - start_time
                )
            
            approach = tool_decision.get("approach")
            tool_name = tool_decision.get("tool")
            tool_inputs = tool_decision.get("inputs", {})
            reasoning = tool_decision.get("reasoning", "")
            
            logger.info(
                "engineer_approach",
                approach=approach,
                tool=tool_name,
                reasoning=reasoning
            )
            
            # Execute the tool
            # Note: In real implementation, import and execute actual tools
            # For now, return a structured success
            
            output = f"ENGINEERING EXECUTION\n"
            output += f"Approach: {approach}\n"
            output += f"Tool: {tool_name}\n"
            output += f"Reasoning: {reasoning}\n\n"
            output += f"[Tool execution would happen here with real tools]\n"
            output += f"Inputs: {json.dumps(tool_inputs, indent=2)}"
            
            duration = time.time() - start_time
            
            logger.info(
                "engineer_completed",
                approach=approach,
                duration=duration
            )
            
            self.record_success()
            
            return AgentResult(
                success=True,
                output=output,
                metadata={
                    "approach": approach,
                    "tool": tool_name,
                    "reasoning": reasoning
                },
                confidence=0.85,
                agent_name=self.name,
                duration_sec=duration
            )
        
        except Exception as e:
            logger.error("engineer_error", error=str(e))
            self.record_failure()
            
            return AgentResult(
                success=False,
                output="",
                errors=[str(e)],
                confidence=0.0,
                agent_name=self.name,
                duration_sec=time.time() - start_time
            )
    
    async def _decide_approach(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to decide engineering approach and generate tool inputs.
        
        Returns:
            Dict with approach, tool, inputs, and reasoning
        """
        context_str = ""
        if context:
            context_str = f"\n\nCONTEXT:\n{json.dumps(context, indent=2)}"
        
        user_prompt = f"""ENGINEERING TASK:
{task}
{context_str}

Determine the best engineering approach.
Return JSON only."""
        
        try:
            response = await call_llm(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.strip()
            
            # Handle markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1] if "\n" in response_text else response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
            
            # Extract JSON
            start = response_text.find("{")
            end = response_text.rfind("}")
            
            if start != -1 and end != -1:
                response_text = response_text[start:end+1]
            
            decision = json.loads(response_text)
            
            logger.debug(
                "engineer_decision",
                approach=decision.get("approach"),
                tool=decision.get("tool")
            )
            
            return decision
        
        except Exception as e:
            logger.error("engineer_decision_error", error=str(e))
            return None