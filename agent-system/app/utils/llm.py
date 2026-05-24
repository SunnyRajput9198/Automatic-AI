import os
from groq import Groq
import asyncio
import structlog
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from anthropic.types import TextBlock  # add this import

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
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        async with self._lock:
            now = datetime.now()

            self.calls = [
                t for t in self.calls
                if now - t < self.period
            ]

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

            self.calls.append(datetime.now())


rate_limiter = RateLimiter(max_calls=5, period_seconds=60)


# -------------------------------
# Groq Client (Initialized Once)
# ---------------------
_groq_api_key = os.getenv("GROQ_API_KEY")
if not _groq_api_key:
    logger.warning("GROQ_API_KEY not set - Groq calls will fail")

_groq_client = Groq(api_key=_groq_api_key) if _groq_api_key else None

# Groq default model
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

# Separate rate limiter for Groq (generous free tier)
groq_rate_limiter = RateLimiter(max_calls=25, period_seconds=60)

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
    max_tokens: int,
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
    # remove the old: return response.content[0].text
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
    
async def call_llm_with_system(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> str:
    """
    Convenience wrapper for the common system+user message pattern.
    Used by most agents — reduces boilerplate from 8 lines to 1.
    """
    return await call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
def _sync_groq_call(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    if not _groq_client:
        raise RuntimeError("Groq client not initialized - GROQ_API_KEY missing")

    response = _groq_client.chat.completions.create(
        model=model,
        messages=messages,  # Groq uses OpenAI format directly
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def call_groq(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> str:
    """
    Groq LLM call — fast, free tier available.
    Used for: critic, planner, reflection agents.
    Models: llama-3.1-8b-instant, llama-3.3-70b-versatile
    """
    await groq_rate_limiter.wait_if_needed()

    logger.info(
        "groq_call_started",
        model=model,
        num_messages=len(messages),
    )

    try:
        content = await asyncio.to_thread(
            _sync_groq_call,
            messages,
            model,
            temperature,
            max_tokens,
        )

        logger.info(
            "groq_call_completed",
            model=model,
            response_length=len(content),
        )

        return content

    except Exception as e:
        logger.error(
            "groq_call_failed",
            model=model,
            error=str(e),
        )
        raise RuntimeError(f"Groq LLM call failed: {str(e)}")


async def call_groq_with_system(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> str:
    """Convenience wrapper for Groq with system+user pattern."""
    return await call_groq(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )