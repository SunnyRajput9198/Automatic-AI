
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import asyncio
import structlog
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from anthropic.types import TextBlock 
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
logger = structlog.get_logger()

# -------------------------------
# Configuration
# -------------------------------

DEFAULT_MODEL = "claude-haiku-4-5-20251001"  # Best balance for agents


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
_api_key = os.getenv("ANTHROPIC_API_KEY")
if not _api_key:
    raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

_client = Anthropic(api_key=_api_key)

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

def _convert_messages(messages: List[Dict[str, str]]):
    """
    Convert OpenAI-style messages to Anthropic messages

    Returns:
        (system_prompt, anthropic_messages)
    """
    converted = []
    system_prompt = None

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            system_prompt = content
        elif role == "user":
            converted.append({"role": "user", "content": content})
        elif role == "assistant":
            converted.append({"role": "assistant", "content": content})

    return system_prompt, converted


def _sync_claude_call(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int
) -> str:
    system_prompt, anthropic_messages = _convert_messages(messages)

    if system_prompt is not None:
        response = _client.messages.create(
            model=model,
            system=system_prompt,
            messages=anthropic_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        response = _client.messages.create(
            model=model,
            messages=anthropic_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    for block in response.content:
        if isinstance(block, TextBlock):
            return block.text

    raise ValueError("No TextBlock found in Claude response")


# -------------------------------
# Public Async API
# -------------------------------


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