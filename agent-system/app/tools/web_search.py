import asyncio
import time
import structlog
from enum import Enum
from typing import Dict, Any, List, Optional

import httpx
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from app.tools.base import Tool, ToolResult

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Async Circuit Breaker
# ---------------------------------------------------------------------------

class _CircuitState(str, Enum):
    CLOSED   = "closed"    # normal — requests go through
    OPEN     = "open"      # tripped — requests blocked immediately
    HALF_OPEN = "half_open" # testing — one probe request allowed


class CircuitBreaker:
    """
    Async circuit breaker for external HTTP calls.

    States:
      CLOSED   → requests pass through normally.
      OPEN     → all requests fail-fast for `recovery_timeout` seconds.
      HALF_OPEN→ one probe is allowed; success → CLOSED, failure → OPEN again.

    Parameters:
      failure_threshold  — consecutive failures before tripping (default 3)
      recovery_timeout   — seconds to wait before probing again (default 60)
      name               — used in log messages
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout:  float = 60.0,
    ):
        self.name              = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout

        self._state            = _CircuitState.CLOSED
        self._failure_count    = 0
        self._opened_at:  Optional[float] = None
        self._lock             = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def call(self, coro):
        """
        Execute `coro` through the circuit breaker.
        Returns the coroutine's result or raises on open/failure.
        """
        async with self._lock:
            if self._state == _CircuitState.OPEN:
                if time.monotonic() - self._opened_at >= self.recovery_timeout:  # type: ignore[operator]
                    self._state = _CircuitState.HALF_OPEN
                    logger.info("circuit_breaker_half_open", name=self.name)
                else:
                    remaining = round(
                        self.recovery_timeout - (time.monotonic() - self._opened_at), 1  # type: ignore[operator]
                    )
                    logger.warning(
                        "circuit_breaker_open_reject",
                        name=self.name,
                        retry_in_sec=remaining,
                    )
                    raise CircuitOpenError(
                        f"{self.name} circuit is OPEN. Retry in {remaining}s."
                    )

        # Execute outside the lock so other coroutines can proceed
        try:
            result = await coro
            await self._on_success()
            return result
        except CircuitOpenError:
            raise
        except Exception as exc:
            await self._on_failure(exc)
            raise

    @property
    def state(self) -> _CircuitState:
        return self._state

    # ------------------------------------------------------------------
    # Internal state transitions
    # ------------------------------------------------------------------

    async def _on_success(self):
        async with self._lock:
            if self._state == _CircuitState.HALF_OPEN:
                logger.info("circuit_breaker_closed", name=self.name)
            self._state         = _CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at     = None

    async def _on_failure(self, exc: Exception):
        async with self._lock:
            self._failure_count += 1
            logger.warning(
                "circuit_breaker_failure",
                name=self.name,
                count=self._failure_count,
                threshold=self.failure_threshold,
                error=str(exc),
            )
            if self._failure_count >= self.failure_threshold:
                self._state     = _CircuitState.OPEN
                self._opened_at = time.monotonic()
                logger.error(
                    "circuit_breaker_tripped",
                    name=self.name,
                    recovery_in_sec=self.recovery_timeout,
                )


class CircuitOpenError(Exception):
    """Raised when a call is blocked because the circuit is OPEN."""


# ---------------------------------------------------------------------------
# HTTP layer — with circuit breakers + exponential backoff
# ---------------------------------------------------------------------------

class _SemanticScholarAPI:
    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS   = "title,abstract,authors,citationCount,url,externalIds"

    # One breaker per class (shared across all instances — process-level)
    _breaker = CircuitBreaker(
        name="semantic_scholar",
        failure_threshold=3,
        recovery_timeout=60.0,
    )

    async def search(self, query: str, limit: int = 3) -> List[dict]:
        params = {"query": query, "limit": limit, "fields": self.FIELDS}
        try:
            return await self._breaker.call(self._fetch(params, query))
        except CircuitOpenError as e:
            logger.warning("semantic_scholar_circuit_open", query=query, reason=str(e))
            return []
        except Exception as e:
            logger.error("semantic_scholar_error", error=str(e), query=query)
            return []

    async def _fetch(self, params: dict, query: str) -> List[dict]:
        last_exc: Optional[Exception] = None
        for attempt, backoff in enumerate([0, 1, 2], start=1):
            if backoff:
                await asyncio.sleep(backoff)
            try:
                async with httpx.AsyncClient(
                    timeout=10, headers={"User-Agent": "ResearchAgent/1.0"}
                ) as client:
                    resp = await client.get(self.BASE_URL, params=params)

                if resp.status_code == 429:
                    # Rate-limited — treat as a retriable failure with longer wait
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning(
                        "semantic_scholar_rate_limited",
                        retry_after=retry_after,
                        attempt=attempt,
                    )
                    await asyncio.sleep(retry_after)
                    last_exc = Exception(f"HTTP 429 (rate limited)")
                    continue

                if resp.status_code != 200:
                    raise Exception(f"HTTP {resp.status_code}")

                papers = []
                for p in resp.json().get("data", []):
                    authors = ", ".join(
                        a.get("name", "") for a in (p.get("authors") or [])
                    )
                    url = p.get("url") or ""
                    if not url and p.get("paperId"):
                        url = f"https://www.semanticscholar.org/paper/{p['paperId']}"
                    papers.append({
                        "source":         "Semantic Scholar",
                        "title":          p.get("title", ""),
                        "content":        (p.get("abstract") or "")[:500],
                        "authors":        authors,
                        "citation_count": p.get("citationCount", 0),
                        "url":            url,
                    })
                return papers

            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "semantic_scholar_timeout",
                    attempt=attempt,
                    query=query[:60],
                )
                continue

        raise last_exc or Exception("Semantic Scholar: all retries exhausted")


class _WikipediaRESTAPI:
    SEARCH_URL  = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    # One breaker per class (shared across all instances — process-level)
    _breaker = CircuitBreaker(
        name="wikipedia",
        failure_threshold=3,
        recovery_timeout=60.0,
    )

    async def search(self, query: str) -> List[dict]:
        try:
            return await self._breaker.call(self._fetch(query))
        except CircuitOpenError as e:
            logger.warning("wikipedia_circuit_open", query=query, reason=str(e))
            return []
        except Exception as e:
            logger.error("wikipedia_rest_error", error=str(e), query=query)
            return []

    async def _fetch(self, query: str) -> List[dict]:
        last_exc: Optional[Exception] = None
        # Exponential backoff: 0s, 1s, 2s
        for attempt, backoff in enumerate([0, 1, 2], start=1):
            if backoff:
                await asyncio.sleep(backoff)
            try:
                async with httpx.AsyncClient(
                    timeout=10, headers={"User-Agent": "ResearchAgent/1.0"}
                ) as client:
                    # Step 1: find best matching title
                    search_resp = await client.get(self.SEARCH_URL, params={
                        "action": "query", "list": "search",
                        "srsearch": query, "srlimit": 1, "format": "json",
                    })

                    if search_resp.status_code == 429:
                        retry_after = int(search_resp.headers.get("Retry-After", 5))
                        logger.warning(
                            "wikipedia_rate_limited",
                            retry_after=retry_after,
                            attempt=attempt,
                        )
                        await asyncio.sleep(retry_after)
                        last_exc = Exception("HTTP 429 (rate limited)")
                        continue

                    if search_resp.status_code != 200:
                        raise Exception(f"HTTP {search_resp.status_code} on search")

                    hits = search_resp.json().get("query", {}).get("search", [])
                    if not hits:
                        return []

                    title = hits[0]["title"]

                    # Step 2: fetch page summary
                    summary_resp = await client.get(
                        self.SUMMARY_URL.format(title=title.replace(" ", "_"))
                    )
                    if summary_resp.status_code != 200:
                        raise Exception(
                            f"HTTP {summary_resp.status_code} on summary for '{title}'"
                        )

                    page = summary_resp.json()
                    canonical_url = (
                        page.get("content_urls", {}).get("desktop", {}).get("page", "")
                        or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    )
                    return [{
                        "source":  "Wikipedia",
                        "title":   page.get("title", title),
                        "content": (page.get("extract") or "")[:500],
                        "url":     canonical_url,
                    }]

            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "wikipedia_timeout",
                    attempt=attempt,
                    query=query[:60],
                )
                continue

        raise last_exc or Exception("Wikipedia: all retries exhausted")


# ---------------------------------------------------------------------------
# Standalone bound tools
# ---------------------------------------------------------------------------

class SemanticScholarTool(Tool):
    """
    Search Semantic Scholar for academic papers.
    Use this for queries about research papers, ML models, algorithms,
    benchmarks, authors, or any academic/scientific topic.
    """

    def __init__(self):
        self._api = _SemanticScholarAPI()

    @property
    def name(self) -> str:
        return "semantic_scholar_search"

    @property
    def description(self) -> str:
        return (
            "Search Semantic Scholar for academic papers. "
            "Returns title, abstract, authors, citation count, and URL. "
            "Best for: research papers, ML models, algorithms, scientific studies."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Academic search query (e.g. 'attention mechanism transformer')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of papers to return (default: 3, max: 10)",
                    "default": 3,
                },
            },
            "required": ["query"],
        }

    async def run(self, **kwargs) -> ToolResult:
        query: str = kwargs.get("query", "")
        limit: int = min(kwargs.get("limit", 3), 10)

        if not query.strip():
            return ToolResult(success=False, output="", error="Query is required")

        logger.info("semantic_scholar_search_running", query=query, limit=limit)
        papers = await self._api.search(query, limit=limit)

        if not papers:
            return ToolResult(
                success=False, output="", error="No papers found",
                metadata={"query": query, "num_results": 0},
            )

        formatted = []
        for i, p in enumerate(papers, 1):
            formatted.append(
                f"{i}. [Semantic Scholar] {p['title']}\n"
                f"   {p['content']}\n"
                f"   Authors: {p['authors']}  |  Citations: {p['citation_count']}\n"
                f"   URL: {p['url']}"
            )

        logger.info("semantic_scholar_search_completed", num_results=len(papers))
        return ToolResult(
            success=True,
            output="\n\n".join(formatted),
            metadata={
                "tool_name":   "semantic_scholar_search",
                "query":       query,
                "num_results": len(papers),
                "sources":     ["Semantic Scholar"],
            },
        )


class WikipediaTool(Tool):
    """
    Look up a topic on Wikipedia using the REST API.
    Use this for factual, encyclopedic, or general-knowledge queries.
    """

    def __init__(self):
        self._api = _WikipediaRESTAPI()

    @property
    def name(self) -> str:
        return "wikipedia_search"

    @property
    def description(self) -> str:
        return (
            "Search Wikipedia for a topic. "
            "Returns the page title, a concise summary, and the canonical URL. "
            "Best for: factual lookups, definitions, historical context, general knowledge."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or question to look up on Wikipedia",
                },
            },
            "required": ["query"],
        }

    async def run(self, **kwargs) -> ToolResult:
        query: str = kwargs.get("query", "")

        if not query.strip():
            return ToolResult(success=False, output="", error="Query is required")

        logger.info("wikipedia_search_running", query=query)
        results = await self._api.search(query)

        if not results:
            return ToolResult(
                success=False, output="", error="No Wikipedia article found",
                metadata={"query": query, "num_results": 0},
            )

        p = results[0]
        output = (
            f"1. [Wikipedia] {p['title']}\n"
            f"   {p['content']}\n"
            f"   URL: {p['url']}"
        )

        logger.info("wikipedia_search_completed", title=p["title"])
        return ToolResult(
            success=True,
            output=output,
            metadata={
                "tool_name":   "wikipedia_search",
                "query":       query,
                "num_results": 1,
                "sources":     ["Wikipedia"],
            },
        )


# ---------------------------------------------------------------------------
# WebSearchTool — combined fallback for ambiguous / general queries
# ---------------------------------------------------------------------------

class WebSearchTool(Tool):
    """
    Multi-source web search combining Semantic Scholar, Wikipedia, and DuckDuckGo.
    Use this when the query type is ambiguous or you want broad coverage.
    For academic queries prefer semantic_scholar_search.
    For factual lookups prefer wikipedia_search.
    """

    _ACADEMIC_KW = [
        "neural network", "algorithm", "paper", "model", "architecture",
        "theorem", "proof", "dataset", "benchmark", "method", "approach",
        "deep learning", "machine learning", "research", "study",
        "transformer", "attention mechanism", "gradient", "optimization",
        "loss function", "embedding", "inference", "training",
    ]
    _PRODUCT_KW = [
        "fastapi", "django", "flask", "react", "vue", "angular",
        "docker", "kubernetes", "asyncio", "nodejs", "express",
        "postgresql", "mongodb", "redis", "graphql", "rest api",
        "next.js", "svelte", "tailwind",
    ]
    _TECHNICAL_KW = [
        "implementation", "how to", "tutorial", "example", "code",
        "library", "framework", "api", "sdk", "install", "configure",
    ]

    def __init__(self):
        self._ss   = _SemanticScholarAPI()
        self._wiki = _WikipediaRESTAPI()
        self._ddg  = DuckDuckGoSearchAPIWrapper(
            region="wt-wt", safesearch="moderate", time="y", max_results=5
        )

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Broad web search across Semantic Scholar, Wikipedia, and DuckDuckGo. "
            "Use for ambiguous or general queries. "
            "For academic papers use semantic_scholar_search. "
            "For factual lookups use wikipedia_search."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def _is_academic(self, q: str) -> bool:
        if any(kw in q for kw in self._PRODUCT_KW):
            return False
        return any(kw in q for kw in self._ACADEMIC_KW)

    def _is_technical(self, q: str) -> bool:
        return any(kw in q for kw in self._TECHNICAL_KW + self._ACADEMIC_KW)

    async def _ddg_search(self, query: str) -> List[dict]:
        try:
            items = await asyncio.to_thread(self._ddg.results, query, max_results=5)
            return [{
                "source":  "DuckDuckGo",
                "title":   i.get("title", ""),
                "content": i.get("snippet", "")[:500],
                "url":     i.get("link", ""),
            } for i in items]
        except Exception as e:
            logger.error("duckduckgo_search_error", error=str(e))
            return []

    async def run(self, **kwargs) -> ToolResult:
        query: str       = kwargs.get("query", "")
        max_results: int = kwargs.get("max_results", 5)

        if not query.strip():
            return ToolResult(success=False, output="", error="Search query is required")

        logger.info("web_search_running", query=query)
        q = query.lower()

        try:
            if self._is_academic(q) or self._is_technical(q):
                strategy = "semantic_scholar+wikipedia+duckduckgo"
                results = await asyncio.gather(
                    self._ss.search(query),
                    self._wiki.search(query),
                    self._ddg_search(query),
                    return_exceptions=True,
                )
            else:
                strategy = "wikipedia+duckduckgo"
                results = await asyncio.gather(
                    self._wiki.search(query),
                    self._ddg_search(query),
                    return_exceptions=True,
                )

            logger.info("web_search_strategy", strategy=strategy)

            sources: List[dict] = [
                item for r in results if isinstance(r, list) for item in r
            ]

            if not sources:
                return ToolResult(
                    success=False, output="", error="No results found",
                    metadata={"query": query, "num_results": 0},
                )

            formatted = []
            for i, s in enumerate(sources[:max_results], 1):
                extra = ""
                if s.get("authors"):
                    extra += f"\n   Authors: {s['authors']}"
                if s["source"] == "Semantic Scholar":
                    extra += f"  |  Citations: {s.get('citation_count', 0)}"
                formatted.append(
                    f"{i}. [{s['source']}] {s['title']}\n"
                    f"   {s.get('content', '')[:500]}"
                    f"{extra}\n"
                    f"   URL: {s.get('url', '')}"
                )

            source_names = list({s["source"] for s in sources})
            logger.info("web_search_completed", num_results=len(sources), sources=source_names)

            return ToolResult(
                success=True,
                output="\n\n".join(formatted),
                metadata={
                    "tool_name":   "web_search",
                    "query":       query,
                    "num_results": len(sources),
                    "sources":     source_names,
                },
            )

        except Exception as e:
            logger.error("web_search_error", error=str(e))
            return ToolResult(success=False, output="", error=f"Search failed: {str(e)}")


# ---------------------------------------------------------------------------
# WebFetchTool — unchanged
# ---------------------------------------------------------------------------

class WebFetchTool(Tool):
    """Fetch text content from a URL (HTTPS only)."""

    BLOCKED_DOMAINS = ["localhost", "127.0.0.1", "0.0.0.0", "internal", "private"]

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch text content from a webpage (HTTPS only)"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to fetch"}},
            "required": ["url"],
        }

    async def run(self, **kwargs) -> ToolResult:
        url: str = kwargs.get("url", "")

        if not url:
            return ToolResult(success=False, output="", error="URL is required")
        if not url.startswith(("https://", "http://")):
            return ToolResult(success=False, output="", error="URL must start with http:// or https://")
        if any(b in url.lower() for b in self.BLOCKED_DOMAINS):
            return ToolResult(success=False, output="", error="Access to internal domains not allowed")

        logger.info("web_fetch_running", url=url)

        try:
            async with httpx.AsyncClient(
                timeout=15.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    return ToolResult(success=False, output="", error=f"HTTP {resp.status_code}")

                content = resp.text[:50000]
                logger.info("web_fetch_completed", url=url, size=len(content))
                return ToolResult(
                    success=True,
                    output=content,
                    metadata={
                        "tool_name":   "web_fetch",
                        "url":         url,
                        "status_code": resp.status_code,
                        "size":        len(content),
                    },
                )
        except Exception as e:
            logger.error("web_fetch_error", error=str(e), url=url)
            return ToolResult(success=False, output="", error=f"Fetch failed: {str(e)}")
