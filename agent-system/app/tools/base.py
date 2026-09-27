from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class ToolResult(BaseModel):
    """Standardized tool execution result"""

    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def classify_tool_failure(error: Optional[str], status_code: Optional[int] = None) -> str:
    """Normalize common upstream/tool errors without storing sensitive payloads."""
    message = (error or "").lower()
    if status_code in (401, 403) or "unauthorized" in message or "authentication" in message:
        return "AUTHENTICATION_FAILURE"
    if status_code == 429 or "rate limit" in message or "quota" in message:
        return "RATE_LIMIT"
    if status_code in (408, 504) or "timeout" in message or "timed out" in message:
        return "TIMEOUT"
    if status_code and status_code >= 500 or "http 5" in message or "provider error" in message:
        return "PROVIDER_5XX"
    if any(word in message for word in ("connect", "network", "dns", "ssl", "handshake")):
        return "NETWORK_ERROR"
    if "unsupported" in message or "not supported" in message:
        return "UNSUPPORTED_OPERATION"
    if "no results" in message or "not found" in message or "empty" in message:
        return "EMPTY_RESULT"
    if status_code in (400, 422) or "invalid request" in message:
        return "INVALID_REQUEST"
    return "TOOL_ERROR"


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
    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Convert this tool to the OpenAI function-calling schema format.
        Used for real tool binding via the `tools=` API parameter.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def validate_input(self, **kwargs) -> bool:
        """Validate input against schema (basic validation)"""
        required_keys = self.input_schema.get("required", [])

        for key in required_keys:
            if key not in kwargs:
                logger.error("tool_missing_param", tool=self.name, param=key)
                return False

        return True
