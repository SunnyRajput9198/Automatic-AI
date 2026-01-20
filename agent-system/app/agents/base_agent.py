from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()


class AgentResult(BaseModel):
    """Standard result format from any agent"""
    success: bool
    output: str
    metadata: Dict[str, Any] = {}
    confidence: float = 0.5  # How confident is this agent in the result
    errors: List[str] = []
    agent_name: str
    duration_sec: float = 0.0


class BaseAgent(ABC):
    """
    Base class for all specialist agents.
    
    Each agent has:
    - A specific role (researcher, engineer, writer)
    - Limited tool access (enforced by allowed_tools)
    - Consistent output format (AgentResult)
    
    This enables:
    - Easy agent swapping
    - Clear responsibility boundaries
    - Reliable coordinator integration
    """
    
    def __init__(self, name: str, role: str, allowed_tools: List[str]):
        """
        Initialize base agent.
        
        Args:
            name: Agent identifier (e.g., "researcher_001")
            role: Agent role (e.g., "researcher", "engineer")
            allowed_tools: List of tool names this agent can use
        """
        self.name = name
        self.role = role
        self.allowed_tools = allowed_tools
        self.call_count = 0
        self.success_count = 0
        self.failure_count = 0
        
        logger.info(
            "agent_initialized",
            name=self.name,
            role=self.role,
            allowed_tools=self.allowed_tools
        )
    
    @abstractmethod
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute the task using this agent's capabilities.
        
        Args:
            task: Task description
            context: Optional context from previous steps or coordinator
            
        Returns:
            AgentResult with output and metadata
        """
        pass
    
    def can_use_tool(self, tool_name: str) -> bool:
        """Check if this agent is allowed to use a specific tool"""
        return tool_name in self.allowed_tools
    
    def record_success(self):
        """Track successful execution"""
        self.call_count += 1
        self.success_count += 1
        
    def record_failure(self):
        """Track failed execution"""
        self.call_count += 1
        self.failure_count += 1
    
    def get_success_rate(self) -> float:
        """Calculate agent's success rate"""
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return {
            "name": self.name,
            "role": self.role,
            "calls": self.call_count,
            "successes": self.success_count,
            "failures": self.failure_count,
            "success_rate": self.get_success_rate()
        }