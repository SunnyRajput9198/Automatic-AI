from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()


class ToolResult(BaseModel):
    """Standardized tool execution result"""

    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


# ABC = Abstract Base Class
class Tool(ABC):
    """
    Base class for all tools.

    Every tool MUST implement:
    - name: Unique identifier
    - description: What the tool does
    - input_schema: Expected input parameters
    - run: Execution logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description"""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON schema for tool inputs"""
        pass

    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        """
        Execute the tool

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with success status and output
        """
        pass

    # @abstractmethod is used inside an abstract base class to declare methods that must be implemented by subclasses.
    # A class containing abstract methods cannot be instantiated until all abstract methods are overridden by the child class.
    # @property allows a method to be accessed like an attribute.
    # It provides a clean interface while still executing code behind the scenes. Instead of calling obj.method(), you can access it as obj.property.
    def validate_input(self, **kwargs) -> bool:
        """Validate input against schema (basic validation)"""
        required_keys = self.input_schema.get("required", [])

        for key in required_keys:
            if key not in kwargs:
                logger.error("tool_missing_param", tool=self.name, param=key)
                return False

        return True
