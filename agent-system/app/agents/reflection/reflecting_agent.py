import structlog
from typing import Dict, Any, Optional
from pydantic import BaseModel

from app.agents.base_agent import BaseAgent, AgentResult
from app.utils.llm import call_llm

logger = structlog.get_logger()


class ReflectionResult(BaseModel):
    failure_reason: str
    suggested_action: str
    confidence: float


class ReflectionAgent(BaseAgent):
    """
    Analyzes failures and suggests recovery actions.

    This agent NEVER executes tools.
    It only reasons about failures.
    """

    SYSTEM_PROMPT = """
You are a failure analysis AI agent.

Your job:
- Analyze why an agent or tool failed
- Suggest the best recovery strategy

Common strategies:
- retry_with_smaller_prompt
- retry_after_delay
- switch_agent
- skip_step
- continue_without_this_step

Return JSON only.

Format:
{
  "failure_reason": "...",
  "suggested_action": "...",
  "confidence": 0.0
}
"""

    def __init__(self, name: str = "reflection_001"):
        super().__init__(
            name=name,
            role="reflection",
            allowed_tools=[]
        )

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:

        logger.info("reflection_started")

        try:
            context_str = ""
            if context:
                context_str = f"\n\nCONTEXT:\n{context}"

            user_prompt = f"""
TASK:
{task}

FAILURE CONTEXT:
{context_str}

Analyze why this failed and suggest recovery.
"""

            response = await call_llm(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )

            return AgentResult(
                success=True,
                output=response,
                confidence=0.9,
                agent_name=self.name,
            )

        except Exception as e:
            logger.error("reflection_failed", error=str(e))

            return AgentResult(
                success=False,
                output="",
                errors=[str(e)],
                confidence=0.0,
                agent_name=self.name,
            )
