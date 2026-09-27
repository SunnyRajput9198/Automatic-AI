import os
import re
import time
import asyncio
import structlog
from contextvars import ContextVar, Token
from contextlib import contextmanager
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.core.config import settings

load_dotenv()
logger = structlog.get_logger()

_llm_attempt_sink: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar("llm_attempt_sink", default=None)
_current_llm_attempt: ContextVar[Optional[Dict[str, Any]]] = ContextVar("current_llm_attempt", default=None)


@contextmanager
def capture_llm_attempts():
    """Collect sanitized call metadata in the current async context."""
    attempts: List[Dict[str, Any]] = []
    token: Token = _llm_attempt_sink.set(attempts)
    try:
        yield attempts
    finally:
        _llm_attempt_sink.reset(token)


def _store_llm_attempt(record: Dict[str, Any]) -> None:
    sink = _llm_attempt_sink.get()
    if sink is not None:
        sink.append(dict(record))

DEFAULT_MODEL = settings.DEFAULT_ANTHROPIC_MODEL
DEFAULT_OPENAI_MODEL = settings.DEFAULT_OPENAI_MODEL


# -------------------------------
# Dynamic Config Resolvers
# -------------------------------

def _get_openai_api_key() -> str:
    key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not configured in settings or environment")
    return key


def _get_openai_base_url() -> str:
    return settings.OPENAI_BASE_URL or os.getenv("OPENAI_BASE_URL", "https://aicredits.in/v1")


def _get_anthropic_api_key() -> str:
    key = settings.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured in settings or environment")
    return key


# -------------------------------
# Error Sanitization & Diagnostics
# -------------------------------

def sanitize_error(error_msg: str) -> str:
    """Mask potential API keys or tokens in error strings."""
    if not error_msg:
        return "Unknown error"
    # Mask OpenAI / Anthropic key formats
    sanitized = re.sub(r"sk-[a-zA-Z0-9\-_]{15,}", "sk-***", str(error_msg))
    return sanitized


def classify_llm_exception(e: Exception) -> Dict[str, Any]:
    """Categorize LLM exception into actionable failure modes."""
    causes = []
    current = e
    while current is not None and current not in causes:
        causes.append(current)
        current = current.__cause__ or current.__context__

    err_str = " ".join(str(error).lower() for error in causes)
    status_code = next(
        (
            getattr(error, "status_code", getattr(getattr(error, "response", None), "status_code", None))
            for error in causes
            if getattr(error, "status_code", getattr(getattr(error, "response", None), "status_code", None)) is not None
        ),
        None,
    )

    if "empty response from openai via langchain" in err_str:
        return {"category": "EMPTY_RESPONSE_ERROR", "retryable": True, "status_code": status_code}
    if "401" in err_str or "unauthorized" in err_str or "invalid_api_key" in err_str or status_code in (401, 403):
        return {"category": "AUTHENTICATION_ERROR", "retryable": False, "status_code": status_code or 401}
    elif "429" in err_str or "rate limit" in err_str or status_code == 429:
        return {"category": "RATE_LIMIT_ERROR", "retryable": True, "status_code": status_code or 429}
    elif "timeout" in err_str or "timed out" in err_str or status_code in (408, 504):
        return {"category": "TIMEOUT_ERROR", "retryable": True, "status_code": status_code or 504}
    elif "400" in err_str or "bad request" in err_str or status_code == 400:
        return {"category": "INVALID_REQUEST_ERROR", "retryable": False, "status_code": status_code or 400}
    elif any(code in err_str for code in ("500", "502", "503", "connection")) or (status_code and status_code >= 500):
        return {"category": "PROVIDER_ERROR", "retryable": True, "status_code": status_code or 500}
    else:
        return {"category": "UNKNOWN_LLM_ERROR", "retryable": True, "status_code": status_code or 500}


# -------------------------------
# Simple Async Rate Limiter
# -------------------------------

class RateLimiter:
    def __init__(self, max_calls: int = 10, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period = timedelta(seconds=period_seconds)
        self.calls: List[datetime] = []
        self._lock = asyncio.Lock()

    async def wait_if_needed(self):
        async with self._lock:
            now = datetime.now()
            self.calls = [t for t in self.calls if now - t < self.period]
            if len(self.calls) >= self.max_calls:
                oldest = min(self.calls)
                wait_seconds = (oldest + self.period - now).total_seconds()
                if wait_seconds > 0:
                    logger.warning(
                        "rate_limit_waiting",
                        wait_seconds=round(wait_seconds, 2),
                    )
                    await asyncio.sleep(wait_seconds)
                    self.calls = [t for t in self.calls if datetime.now() - t < self.period]
            self.calls.append(datetime.now())


rate_limiter = RateLimiter(max_calls=5, period_seconds=60)
openai_rate_limiter = RateLimiter(max_calls=25, period_seconds=60)


# -------------------------------
# Message Builder Helper
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


# -------------------------------
# Generic Retry Runner
# -------------------------------

async def _call_with_retries(
    func,
    *args,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    provider: str = "ai_credits",
    telemetry_model: str = DEFAULT_OPENAI_MODEL,
    **kwargs,
) -> Any:
    """Run sync function in worker thread with exponential backoff and error classification."""
    start_time = time.time()
    attempt = 0
    last_exception = None

    while attempt < max_retries:
        attempt += 1
        request_kwargs = kwargs
        if not request_kwargs and args and isinstance(args[0], dict):
            request_kwargs = args[0]
        messages = request_kwargs.get("messages", [])
        attempt_record: Dict[str, Any] = {
            "provider": provider,
            "model": telemetry_model,
            "attempt": attempt,
            "started_at": time.time(),
            "base_url": _get_openai_base_url() if provider in ("ai_credits", "openai") else None,
            "input_characters": sum(
                len(str(message.get("content", "")))
                for message in messages if isinstance(message, dict)
            ) if isinstance(messages, list) else None,
            "max_tokens": request_kwargs.get("max_tokens"),
            "temperature": request_kwargs.get("temperature"),
            "reasoning_effort": request_kwargs.get("reasoning_effort"),
        }
        attempt_token = _current_llm_attempt.set(attempt_record)
        try:
            logger.info(
                "llm_call_started",
                provider=provider,
                model=telemetry_model,
                attempt=attempt,
                base_url=_get_openai_base_url() if provider in ("ai_credits", "openai") else None,
            )
            result = await asyncio.to_thread(func, *args, **kwargs)
            latency = round(time.time() - start_time, 3)
            attempt_record.update(status="success", duration_ms=round((time.time() - attempt_record["started_at"]) * 1000, 2))
            _store_llm_attempt(attempt_record)
            logger.info(
                "llm_call_completed",
                provider=provider,
                model=telemetry_model,
                latency_sec=latency,
                attempt=attempt,
            )
            return result
        except Exception as e:
            last_exception = e
            diag = classify_llm_exception(e)
            sanitized = sanitize_error(str(e))
            attempt_record.update(
                status="error",
                duration_ms=round((time.time() - attempt_record["started_at"]) * 1000, 2),
                error_category=diag["category"],
                error_type=type(e).__name__,
            )
            _store_llm_attempt(attempt_record)

            logger.warning(
                "llm_call_failed_attempt",
                provider=provider,
                model=telemetry_model,
                attempt=attempt,
                category=diag["category"],
                status_code=diag["status_code"],
                retryable=diag["retryable"],
                error=sanitized,
            )

            if not diag["retryable"] or attempt >= max_retries:
                logger.error(
                    "llm_call_failed_final",
                    provider=provider,
                    model=telemetry_model,
                    total_attempts=attempt,
                    category=diag["category"],
                    error=sanitized,
                )
                raise RuntimeError(
                    f"LLM call failed ({provider}/{telemetry_model} - {diag['category']}): {sanitized}"
                ) from e

            delay = initial_delay * (2 ** (attempt - 1))
            logger.info("llm_retry_backoff", delay_sec=delay, attempt=attempt)
            await asyncio.sleep(delay)
        finally:
            _current_llm_attempt.reset(attempt_token)

    raise RuntimeError(
        f"LLM call failed after {max_retries} attempts: {sanitize_error(str(last_exception))}"
    )


# -------------------------------
# Anthropic Client Implementation
# -------------------------------

def _sync_claude_call(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    api_key = _get_anthropic_api_key()
    llm = ChatAnthropic(
        model=model,  # type: ignore
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore
        api_key=api_key,
    )
    lc_messages = _build_langchain_messages(messages)
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
    await rate_limiter.wait_if_needed()
    return await _call_with_retries(
        _sync_claude_call,
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider="anthropic",
        telemetry_model=model,
    )


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


# -------------------------------
# OpenAI / AI Credits Implementation
# -------------------------------

def _sync_openai_call(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: Optional[str] = None,
) -> str:
    api_key = _get_openai_api_key()
    base_url = _get_openai_base_url()

    client_options: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": _current_llm_attempt.get().get("reasoning_effort") if _current_llm_attempt.get() else None,
        "api_key": api_key,
        "base_url": base_url,
        "request_timeout": 60.0,
        "max_retries": 1,
    }
    if reasoning_effort is not None:
        client_options["reasoning_effort"] = reasoning_effort
    llm = ChatOpenAI(**client_options)
    lc_messages = _build_langchain_messages(messages)
    response = llm.invoke(lc_messages)
    raw_content = response.content
    if isinstance(raw_content, str):
        content = raw_content
    elif isinstance(raw_content, list):
        # LangChain may represent multimodal/text responses as content blocks.
        content = "".join(
            block.get("text", "") if isinstance(block, dict) and block.get("type", "text") == "text"
            else str(block) if isinstance(block, str) else ""
            for block in raw_content
        )
    else:
        content = str(raw_content) if raw_content is not None else ""

    metadata = getattr(response, "response_metadata", {}) or {}
    usage = getattr(response, "usage_metadata", {}) or {}
    additional = getattr(response, "additional_kwargs", {}) or {}
    metadata = metadata if isinstance(metadata, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    additional = additional if isinstance(additional, dict) else {}
    provider_usage = metadata.get("token_usage")
    provider_usage = provider_usage if isinstance(provider_usage, dict) else {}
    tool_calls = getattr(response, "tool_calls", [])
    tool_calls = tool_calls if isinstance(tool_calls, list) else []
    current_attempt = _current_llm_attempt.get()
    # Keep diagnostics useful while excluding prompts, response bodies,
    # authorization headers, and arbitrary provider metadata.
    response_diagnostics = {
        "provider": "ai_credits",
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": current_attempt.get("reasoning_effort") if current_attempt else None,
        "input_characters": sum(len(str(message.get("content", ""))) for message in messages),
        "content_type": type(raw_content).__name__,
        "content_characters": len(content),
        "finish_reason": metadata.get("finish_reason"),
        "response_model": metadata.get("model_name"),
        "usage": {
            key: usage.get(key)
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if isinstance(usage.get(key), (int, float))
        },
        "provider_token_usage": {
            key: provider_usage.get(key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if isinstance(provider_usage.get(key), (int, float))
        },
        "refusal_present": bool(additional.get("refusal")) if isinstance(additional.get("refusal"), str) else False,
        "tool_call_count": len(tool_calls),
    }
    for field in ("finish_reason", "response_model"):
        if not isinstance(response_diagnostics[field], str):
            response_diagnostics[field] = None
    if current_attempt is not None:
        current_attempt.update(response_diagnostics)
    logger.info("llm_response_received", **response_diagnostics)
    if not content.strip():
        logger.error("llm_empty_response_diagnostics", **response_diagnostics)
        raise ValueError("Empty response from OpenAI via LangChain")
    return content


async def call_openai(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    reasoning_effort: Optional[str] = None,
) -> str:
    await openai_rate_limiter.wait_if_needed()
    return await _call_with_retries(
        _sync_openai_call,
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        provider="ai_credits",
        telemetry_model=model,
    )


async def call_openai_with_system(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
    reasoning_effort: Optional[str] = None,
) -> str:
    return await call_openai(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )


# -------------------------------
# OpenAI Tool Binding Implementation
# -------------------------------

def _sync_openai_call_with_tools(
    messages: List[Dict[str, str]],
    tools: List[Dict],
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict:
    api_key = _get_openai_api_key()
    base_url = _get_openai_base_url()

    from langchain_core.messages import AIMessage as LCAIMessage

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,  # type: ignore
        api_key=api_key,
        base_url=base_url,
        request_timeout=60.0,
        max_retries=1,
    )

    llm_with_tools = llm.bind_tools(tools)
    lc_messages = _build_langchain_messages(messages)
    response: LCAIMessage = llm_with_tools.invoke(lc_messages)  # type: ignore

    if hasattr(response, "tool_calls") and response.tool_calls:
        tc = response.tool_calls[0]
        return {
            "type": "tool_call",
            "name": tc["name"],
            "arguments": tc["args"],
        }

    return {"type": "text", "content": str(response.content)}


async def call_openai_with_tools(
    system_prompt: str,
    user_prompt: str,
    tools: List[Dict],
    model: str = DEFAULT_OPENAI_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4000,
) -> Dict:
    await openai_rate_limiter.wait_if_needed()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return await _call_with_retries(
        _sync_openai_call_with_tools,
        messages=messages,
        tools=tools,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider="ai_credits",
        telemetry_model=model,
    )
