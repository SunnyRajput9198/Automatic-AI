import json
import asyncio
import structlog
from typing import Dict, Any
import httpx
import arxiv as arxiv_lib
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    PubMedAPIWrapper,
    WikipediaAPIWrapper
)
from app.tools.base import Tool, ToolResult
import wikipedia
wikipedia.set_rate_limiting(True)
logger = structlog.get_logger()


class WebSearchTool(Tool):
    """Multi-source web search using LangChain wrappers"""

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10, headers={"User-Agent": "ResearchAgent/1.0"})
        self.wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
        self.arxiv_client = arxiv_lib.Client()
        self.ddg_wrapper = DuckDuckGoSearchAPIWrapper(
            region="wt-wt", safesearch="moderate", time="y", max_results=5
        )
        self.pubmed_wrapper = PubMedAPIWrapper(top_k_results=3, doc_content_chars_max=500)

    async def close(self):
        await self.client.aclose()

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web from Wikipedia, ArXiv, PubMed, and DuckDuckGo"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results (default: 5)", "default": 5},
            },
            "required": ["query"],
        }

    def _is_technical_query(self, query: str) -> bool:
        """Determine if query needs all sources or just Wikipedia"""
        general_kw = ["what is", "who is", "explain", "define"]
        technical_kw = ["research", "paper", "algorithm", "implementation", "arxiv"]
        
        q = query.lower()
        if any(kw in q for kw in technical_kw):
            return True
        if any(kw in q for kw in general_kw):
            return False
        return True

    async def _wiki_search(self, query: str) -> list[dict]:
        for attempt in range(2):
            try:
                results = await asyncio.to_thread(self.wiki_wrapper.run, query)
                if not results or len(results.strip()) < 10:
                    return []
                lines = results.split("\n", 1)
                title = lines[0].strip() if len(lines) > 1 and len(lines[0]) < 100 else query
                content = lines[1].strip() if len(lines) > 1 else results
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                return [{"source": "Wikipedia", "title": title, "content": content[:500], "url": url}]
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(1)
                    continue
                logger.error("wiki_search_error", error=str(e))
                return []
        return []
    async def _arxiv_search(self, query: str) -> list[dict]:
        try:
            def fetch():
                search = arxiv_lib.Search(query=query, max_results=3)
                return list(self.arxiv_client.results(search))

            results = await asyncio.to_thread(fetch)
            papers = []
            for paper in results:
                papers.append({
                    "source": "ArXiv",
                    "title": paper.title,
                    "content": paper.summary[:500],
                    "url": paper.entry_id,
                })
            return papers
        except Exception as e:
            logger.error("arxiv_search_error", error=str(e))
            return []

    async def _pubmed_search(self, query: str) -> list[dict]:
        try:
            results = await asyncio.to_thread(self.pubmed_wrapper.run, query)
            papers = []
            
            for entry in results.split("\n\n"):
                if not entry.strip():
                    continue
                title, summary, url = "", "", ""
                for line in entry.split("\n"):
                    if line.startswith("Title:"):
                        title = line.replace("Title:", "").strip()
                    elif line.startswith("Summary:"):
                        summary = line.replace("Summary:", "").strip()
                    elif "pubmed.ncbi.nlm.nih.gov" in line.lower():
                        url = line.replace("URL:", "").strip()
                
                if title:
                    papers.append({"source": "PubMed", "title": title, "content": summary[:500], "url": url})
            
            return papers
        except Exception as e:
            logger.error("pubmed_search_error", error=str(e))
            return []

    async def _duckduckgo_search(self, query: str) -> list[dict]:
        try:
            results_list = await asyncio.to_thread(self.ddg_wrapper.results, query, max_results=5)
            return [
                {"source": "DuckDuckGo", "title": item.get("title", ""),
                "content": item.get("snippet", "")[:500], "url": item.get("link", "")}
                for item in results_list
            ]
        except Exception as e:
            logger.error("duckduckgo_search_error", error=str(e))
            return []

    async def run(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)

        if not query:
            return ToolResult(success=False, output="", error="Search query is required")

        logger.info("web_search_running", query=query)

        try:
            is_technical = self._is_technical_query(query)

            if is_technical:
                results = await asyncio.gather(
                    self._wiki_search(query),
                    self._arxiv_search(query),
                    self._pubmed_search(query),
                    self._duckduckgo_search(query),
                    return_exceptions=True,
                )
                sources = [item for result in results if isinstance(result, list) for item in result]
            else:
                wiki_results = await self._wiki_search(query)
                if not wiki_results:
                    wiki_results = await self._duckduckgo_search(query)  # 👈 fallback
                sources = wiki_results

            if not sources:
                return ToolResult(
                    success=False, output="", error="No results found",
                    metadata={"query": query, "num_results": 0}
                )

            formatted = [
                f"{i}. {s['source']} — {s['title']}\n   {s['content'][:500]}\n   URL: {s['url']}"
                for i, s in enumerate(sources[:max_results], 1)
            ]

            logger.info("web_search_completed", num_results=len(sources))

            return ToolResult(
                success=True,
                output="\n\n".join(formatted),
                metadata={
                    "tool_name": "web_search",
                    "query": query,
                    "num_results": len(sources),
                    "source": "wikipedia+arxiv+pubmed+duckduckgo" if is_technical else "wikipedia"
                },
            )

        except Exception as e:
            logger.error("web_search_error", error=str(e))
            return ToolResult(success=False, output="", error=f"Search failed: {str(e)}")


class WebFetchTool(Tool):
    """Fetch content from a URL"""

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
        url = kwargs.get("url", "")

        if not url:
            return ToolResult(success=False, output="", error="URL is required")

        if not url.startswith(("https://", "http://")):
            return ToolResult(success=False, output="", error="URL must start with http:// or https://")

        if any(blocked in url.lower() for blocked in self.BLOCKED_DOMAINS):
            return ToolResult(success=False, output="", error="Access to internal domains not allowed")

        logger.info("web_fetch_running", url=url)

        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"}
            ) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    return ToolResult(success=False, output="", error=f"HTTP {response.status_code}")

                content = response.text[:50000]
                logger.info("web_fetch_completed", url=url, size=len(content))

                return ToolResult(
                    success=True,
                    output=content,
                    metadata={
                        "tool_name": "web_fetch",
                        "url": url,
                        "status_code": response.status_code,
                        "size": len(content),
                    },
                )

        except Exception as e:
            logger.error("web_fetch_error", error=str(e), url=url)
            return ToolResult(success=False, output="", error=f"Fetch failed: {str(e)}")
