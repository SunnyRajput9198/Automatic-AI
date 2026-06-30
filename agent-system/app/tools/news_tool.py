"""
NewsSearchTool — real-time news via NewsAPI.org with DuckDuckGo fallback.

Behaviour:
- Primary:  NewsAPI.org /v2/everything  (requires NEWSAPI_KEY in .env)
- Fallback: DuckDuckGoSearchAPIWrapper  (used when key is missing OR
            NewsAPI returns no articles OR the monthly quota is exhausted)

The tool is only called when the LLM decides a task needs current news —
it does not run on every query.
"""

import os
import asyncio
import structlog
from typing import Dict, Any, List

import httpx
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from app.tools.base import Tool, ToolResult
from app.core.config import settings

logger = structlog.get_logger()

# NewsAPI free plan: 100 requests/day, articles up to 1 month old.
# Paid plan removes the age restriction.
_NEWSAPI_URL = "https://newsapi.org/v2/everything"

# DuckDuckGo fallback wrapper (shared instance, no API key needed)
_ddg = DuckDuckGoSearchAPIWrapper(
    region="wt-wt", safesearch="moderate", time="d", max_results=5
)


class NewsSearchTool(Tool):
    """
    Search for real-time news articles.

    Primary source: NewsAPI.org (structured, dated results).
    Fallback:       DuckDuckGo news search (used when NewsAPI key is absent,
                    returns no results, or the quota is exhausted).
    """

    @property
    def name(self) -> str:
        return "news_search"

    @property
    def description(self) -> str:
        return (
            "Search for real-time news articles on any topic. "
            "Returns headline, source, date, and URL. "
            "Use for: latest news, current events, today's headlines, breaking news, "
            "recent developments about a person/company/topic."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "News search query (e.g. 'AI regulation Europe 2024')",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of articles to return (default: 5, max: 10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self, **kwargs) -> ToolResult:
        query: str   = kwargs.get("query", "").strip()
        num_results: int = min(int(kwargs.get("num_results", 5)), 10)

        if not query:
            return ToolResult(success=False, output="", error="Query is required")

        logger.info("news_search_running", query=query, num_results=num_results)

        # Try NewsAPI first if key is configured
        api_key = (settings.NEWSAPI_KEY or "").strip()
        if api_key:
            result = await self._newsapi_search(query, num_results, api_key)
            if result.success:
                return result
            logger.warning(
                "news_search_newsapi_failed",
                error=result.error,
                fallback="duckduckgo",
            )

        # DuckDuckGo fallback
        return await self._ddg_search(query, num_results)

    # ------------------------------------------------------------------
    # NewsAPI source
    # ------------------------------------------------------------------

    async def _newsapi_search(
        self, query: str, num_results: int, api_key: str
    ) -> ToolResult:
        params = {
            "q":        query,
            "pageSize": num_results,
            "language": "en",
            "sortBy":   "publishedAt",
            "apiKey":   api_key,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(_NEWSAPI_URL, params=params)

            if resp.status_code == 401:
                return ToolResult(
                    success=False, output="",
                    error="NewsAPI key is invalid or expired",
                )
            if resp.status_code == 426:
                return ToolResult(
                    success=False, output="",
                    error="NewsAPI quota exhausted — falling back to DuckDuckGo",
                )
            if resp.status_code != 200:
                return ToolResult(
                    success=False, output="",
                    error=f"NewsAPI HTTP {resp.status_code}",
                )

            data      = resp.json()
            articles  = data.get("articles", [])

            if not articles:
                return ToolResult(
                    success=False, output="",
                    error="NewsAPI returned no articles",
                )

            formatted = self._format_newsapi(articles[:num_results])
            logger.info("news_search_newsapi_completed", num_results=len(articles))

            return ToolResult(
                success=True,
                output="\n\n".join(formatted),
                metadata={
                    "tool_name":   "news_search",
                    "source":      "NewsAPI",
                    "query":       query,
                    "num_results": len(articles),
                },
            )

        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

    @staticmethod
    def _format_newsapi(articles: List[dict]) -> List[str]:
        results = []
        for i, a in enumerate(articles, 1):
            source      = (a.get("source") or {}).get("name", "Unknown")
            title       = a.get("title", "")
            description = (a.get("description") or "")[:300]
            date        = (a.get("publishedAt") or "")[:10]   # YYYY-MM-DD
            url         = a.get("url", "")
            results.append(
                f"{i}. [{source}] {title}\n"
                f"   {description}\n"
                f"   Date: {date}\n"
                f"   URL: {url}"
            )
        return results

    # ------------------------------------------------------------------
    # DuckDuckGo fallback
    # ------------------------------------------------------------------

    async def _ddg_search(self, query: str, num_results: int) -> ToolResult:
        try:
            news_query = f"{query} news"
            items = await asyncio.to_thread(
                _ddg.results, news_query, max_results=num_results
            )
            if not items:
                return ToolResult(
                    success=False, output="",
                    error="No news results found from DuckDuckGo",
                )

            formatted = []
            for i, item in enumerate(items[:num_results], 1):
                formatted.append(
                    f"{i}. [DuckDuckGo] {item.get('title', '')}\n"
                    f"   {item.get('snippet', '')[:300]}\n"
                    f"   URL: {item.get('link', '')}"
                )

            logger.info("news_search_ddg_completed", num_results=len(items))
            return ToolResult(
                success=True,
                output="\n\n".join(formatted),
                metadata={
                    "tool_name":   "news_search",
                    "source":      "DuckDuckGo (fallback)",
                    "query":       query,
                    "num_results": len(items),
                },
            )

        except Exception as e:
            logger.error("news_search_ddg_error", error=str(e))
            return ToolResult(success=False, output="", error=f"News search failed: {str(e)}")
