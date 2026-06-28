
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import asyncio
import structlog
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from langchain_anthropic import ChatAnthropic 
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger()

DEFAULT_MODEL = "claude-haiku-4-5-20251001"  


# -------------------------------
# Simple Async Rate Limiter
# -------------------------------


class RateLimiter:
    """
    Simple sliding-window async rate limiter
    """

    def __init__(self, max_calls: int = 10, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period = timedelta(seconds=period_seconds)
        self.calls: List[datetime] = []
        self._lock = asyncio.Lock()#_ means Internal/private by convention

    async def wait_if_needed(self):
        async with self._lock:#Only one async task can enter this block at a time.
            now = datetime.now()

            self.calls = [t for t in self.calls if now - t < self.period]

            if len(self.calls) >= self.max_calls:
                oldest = min(self.calls)
                wait_seconds = (oldest + self.period - now).total_seconds()
                if wait_seconds > 0:
                    logger.warning(
                        "rate_limit_waiting",
                        wait_seconds=round(wait_seconds, 2),
                        provider="anthropic",
                    )
                    await asyncio.sleep(wait_seconds)
                    # Re-prune after sleeping, then append
                    self.calls = [t for t in self.calls if datetime.now() - t < self.period]


            self.calls.append(datetime.now())


rate_limiter = RateLimiter(max_calls=5, period_seconds=60)


# -------------------------------
# LangChain OpenAI Client
# -------------------------------
_openai_api_key = os.getenv("OPENAI_API_KEY")
if not _openai_api_key:
    logger.warning("OPENAI_API_KEY not set - OpenAI calls will fail")

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
openai_rate_limiter = RateLimiter(max_calls=25, period_seconds=60)  # OpenAI has higher rate limits

# -------------------------------
# Anthropic Client (Initialized Once)
# -------------------------------
_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _anthropic_api_key:
    logger.warning("ANTHROPIC_API_KEY not set - Anthropic/Claude calls will fail")

# -------------------------------
# Helpers
# -------------------------------
def _build_langchain_messages(messages: List[Dict[str, str]]):
    """Convert OpenAI-style dicts to LangChain message objects."""
    lc_messages = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages

def _sync_claude_call(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    llm = ChatAnthropic(
        model=model, # type: ignore
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore
        api_key=_anthropic_api_key, 
    )
    lc_messages = _build_langchain_messages(messages)  # reuse same helper
    response = llm.invoke(lc_messages)
    content = str(response.content)
    if not content:
        raise ValueError("Empty response from Claude via LangChain")
    return content

async def call_llm(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> str:
    """
    Anthropic Claude LLM call with:
    - async compatibility
    - rate limiting
    - structured logging
    - proper system prompt handling
    """
    await rate_limiter.wait_if_needed()

    logger.info(
        "llm_call_started",
        provider="anthropic",
        model=model,
        num_messages=len(messages),
    )

    try:
        content = await asyncio.to_thread(
            _sync_claude_call,
            messages,
            model,
            temperature,
            max_tokens,
        )

        logger.info(
            "llm_call_completed",
            provider="anthropic",
            model=model,
            response_length=len(content),
        )

        return content

    except Exception as e:
        logger.error(
            "llm_call_failed",
            provider="anthropic",
            model=model,
            error=str(e),
        )
        raise RuntimeError(f"Anthropic LLM call failed: {str(e)}")

def _sync_openai_call(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    if not _openai_api_key:
        raise RuntimeError("OpenAI client not initialized - OPENAI_API_KEY missing")

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,# type: ignore
        api_key=_openai_api_key,
        base_url="https://aicredits.in/v1",
    )
    lc_messages = _build_langchain_messages(messages)
    response = llm.invoke(lc_messages)
    content = str(response.content)  # ← cast to str
    if not content:
        raise ValueError("Empty response from OpenAI via LangChain")
    return content
async def call_llm_with_system(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> str:
    return await call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def call_openai(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> str:
    await openai_rate_limiter.wait_if_needed()
    logger.info("openai_call_started", model=model, num_messages=len(messages))

    try:
        content = await asyncio.to_thread(
            _sync_openai_call, messages, model, temperature, max_tokens
        )
        logger.info("openai_call_completed", model=model, response_length=len(content))
        return content

    except Exception as e:
        logger.error("openai_call_failed", model=model, error=str(e))
        raise RuntimeError(f"OpenAI LLM call failed: {str(e)}")


async def call_openai_with_system(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> str:
    return await call_openai(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# -------------------------------
# OpenAI Tool Binding
# -------------------------------

def _sync_openai_call_with_tools(
    messages: List[Dict[str, str]],
    tools: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict:
    """
    Call OpenAI with tool schemas bound via the `tools=` parameter.
    Returns a dict with either:
      - {"type": "tool_call", "name": ..., "arguments": {...}}  — LLM chose a tool
      - {"type": "text", "content": ...}                        — LLM returned plain text
    """
    if not _openai_api_key:
        raise RuntimeError("OpenAI client not initialized - OPENAI_API_KEY missing")

    from langchain_core.messages import ToolCall
    from langchain_core.messages import AIMessage as LCAIMessage

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore
        api_key=_openai_api_key,
        base_url="https://aicredits.in/v1",
    )

    # Bind tools so OpenAI returns structured tool_calls
    llm_with_tools = llm.bind_tools(tools)
    lc_messages = _build_langchain_messages(messages)
    response: LCAIMessage = llm_with_tools.invoke(lc_messages)  # type: ignore

    # Check if the model issued a tool call
    if hasattr(response, "tool_calls") and response.tool_calls:
        tc = response.tool_calls[0]
        return {
            "type":      "tool_call",
            "name":      tc["name"],
            "arguments": tc["args"],   # already a dict, no JSON parsing needed
        }

    # Fallback: model returned plain text
    return {"type": "text", "content": str(response.content)}


async def call_openai_with_tools(
    system_prompt: str,
    user_prompt: str,
    tools: List[Dict],
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> Dict:
    """
    Async wrapper for tool-bound OpenAI calls.

    Returns a dict:
      {"type": "tool_call", "name": str, "arguments": dict}
      or
      {"type": "text", "content": str}
    """
    await openai_rate_limiter.wait_if_needed()
    logger.info("openai_tool_call_started", model=model, num_tools=len(tools))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    try:
        result = await asyncio.to_thread(
            _sync_openai_call_with_tools,
            messages, tools, model, temperature, max_tokens,
        )
        logger.info(
            "openai_tool_call_completed",
            model=model,
            result_type=result["type"],
            tool_name=result.get("name"),
        )
        return result

    except Exception as e:
        logger.error("openai_tool_call_failed", model=model, error=str(e))
        raise RuntimeError(f"OpenAI tool-binding call failed: {str(e)}")