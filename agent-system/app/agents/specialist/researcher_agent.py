import re
import asyncio
import structlog
import time
from typing import Dict, Any, List, Optional

from app.agents.base_agent import BaseAgent, AgentResult
from app.tools.base import ToolResult
from app.tools.web_search import (
    SemanticScholarTool,
    WikipediaTool,
    WebSearchTool,
    WebFetchTool,
)
from app.tools.news_tool import NewsSearchTool

logger = structlog.get_logger()

# Fetch full page when total snippets are below this threshold
_FETCH_THRESHOLD = 3

# Keywords that signal a NEWS query → use NewsSearchTool
_NEWS_KW = [
    "news", "headline", "latest", "breaking", "today", "current events",
    "recent", "update", "announcement", "report", "coverage",
]
# Keywords that signal an academic query → include Semantic Scholar
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
]


class ResearcherAgent(BaseAgent):
    """
    Specialist agent for web research.

    Pipeline:
    1. Classify query → academic or general
    2. Dispatch source tools in parallel:
         - Academic: SemanticScholarTool + WikipediaTool  (asyncio.gather)
         - General:  WikipediaTool only
       WebSearchTool is kept as a fallback if both specific tools return nothing.
    3. Merge and deduplicate results across sources
    4. Fetch top URL if total snippets are below _FETCH_THRESHOLD
    5. Return structured findings with per-source breakdown
    """

    def __init__(self, name: str = "researcher_001"):
        super().__init__(
            name=name,
            role="researcher",
            allowed_tools=[
                "semantic_scholar_search",
                "wikipedia_search",
                "news_search",
                "web_search",
                "web_fetch",
            ],
        )
        self.semantic_scholar = SemanticScholarTool()
        self.wikipedia        = WikipediaTool()
        self.news             = NewsSearchTool()
        self.web_search       = WebSearchTool()   # fallback only
        self.web_fetch        = WebFetchTool()

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    def _is_academic(self, query: str) -> bool:
        q = query.lower()
        if any(kw in q for kw in _PRODUCT_KW):
            return False
        return any(kw in q for kw in _ACADEMIC_KW)

    def _is_news(self, query: str) -> bool:
        q = query.lower()
        return any(kw in q for kw in _NEWS_KW)

    # ------------------------------------------------------------------
    # Parallel source dispatch
    # ------------------------------------------------------------------

    async def _parallel_search(self, query: str) -> Dict[str, ToolResult]:
        """
        Fire the appropriate source tools in parallel and return a
        {tool_name: ToolResult} map so each source is tracked independently.

        Routing:
          news query   → news_search (+ wikipedia for context)
          academic     → semantic_scholar + wikipedia
          general      → wikipedia only
        """
        is_news     = self._is_news(query)
        is_academic = self._is_academic(query)

        if is_news:
            logger.info("researcher_parallel_dispatch",
                        sources=["news_search", "wikipedia_search"], query=query)
            news_result, wiki_result = await asyncio.gather(
                self.news.run(query=query, num_results=8),
                self.wikipedia.run(query=query),
                return_exceptions=True,
            )
            return {
                "news_search": (
                    news_result if isinstance(news_result, ToolResult)
                    else ToolResult(success=False, output="", error=str(news_result))
                ),
                "wikipedia_search": (
                    wiki_result if isinstance(wiki_result, ToolResult)
                    else ToolResult(success=False, output="", error=str(wiki_result))
                ),
            }

        if is_academic:
            logger.info("researcher_parallel_dispatch",
                        sources=["semantic_scholar_search", "wikipedia_search"],
                        query=query)
            ss_result, wiki_result = await asyncio.gather(
                self.semantic_scholar.run(query=query, limit=3),
                self.wikipedia.run(query=query),
                return_exceptions=True,
            )
            return {
                "semantic_scholar_search": (
                    ss_result if isinstance(ss_result, ToolResult)
                    else ToolResult(success=False, output="", error=str(ss_result))
                ),
                "wikipedia_search": (
                    wiki_result if isinstance(wiki_result, ToolResult)
                    else ToolResult(success=False, output="", error=str(wiki_result))
                ),
            }

        # General query — Wikipedia only
        logger.info("researcher_parallel_dispatch",
                    sources=["wikipedia_search"], query=query)
        wiki_result = await self.wikipedia.run(query=query)
        return {"wikipedia_search": wiki_result}

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------

    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        start_time = time.time()
        logger.info("researcher_executing", task=task)

        try:
            search_query = self._extract_search_query(task)
            logger.info("researcher_searching", query=search_query)

            # ── Step 1: parallel source dispatch ─────────────────────────
            source_results = await self._parallel_search(search_query)

            successful_sources: List[str] = []
            combined_output_parts: List[str] = []
            num_results = 0

            for source_name, result in source_results.items():
                if result.success and result.output.strip():
                    successful_sources.append(source_name)
                    combined_output_parts.append(result.output)
                    num_results += result.metadata.get("num_results", 1)
                    logger.info("researcher_source_ok",
                                source=source_name,
                                results=result.metadata.get("num_results", 1))
                else:
                    logger.warning("researcher_source_failed",
                                    source=source_name,
                                    error=result.error)

            # ── Step 2: fallback to WebSearchTool if both sources failed ──
            if not combined_output_parts:
                logger.warning("researcher_falling_back_to_web_search", query=search_query)
                fallback = await self.web_search.run(query=search_query, max_results=6)
                if fallback.success:
                    combined_output_parts.append(fallback.output)
                    num_results = fallback.metadata.get("num_results", 0)
                    successful_sources = fallback.metadata.get("sources", ["web_search"])
                else:
                    logger.error("researcher_all_sources_failed", error=fallback.error)
                    self.record_failure()
                    return AgentResult(
                        success=False,
                        output="",
                        errors=[fallback.error or "All search sources failed"],
                        confidence=0.0,
                        agent_name=self.name,
                        duration_sec=time.time() - start_time,
                    )

            combined_output = "\n\n".join(combined_output_parts)

            # ── Step 3: fetch full page if snippets are thin ─────────────
            if num_results < _FETCH_THRESHOLD:
                top_url = self._extract_first_url(combined_output)
                if top_url:
                    logger.info("researcher_fetching_url",
                                url=top_url, reason="thin snippets")
                    fetch_result = await self.web_fetch.run(url=top_url)
                    if fetch_result.success and fetch_result.output:
                        combined_output += (
                            f"\n\nFetched full page from {top_url}:\n"
                            f"{fetch_result.output[:1500]}"
                        )
                        num_results += 1

            # ── Step 4: deduplicate ───────────────────────────────────────
            combined_output = self._deduplicate(combined_output)

            # ── Step 5: format and return ─────────────────────────────────
            output = self._format_research_output(
                query=search_query,
                search_output=combined_output,
                sources=successful_sources,
                num_results=num_results,
            )

            confidence = min(0.95, 0.5 + num_results * 0.1)
            duration   = time.time() - start_time

            logger.info("researcher_completed",
                        num_results=num_results,
                        sources=successful_sources,
                        confidence=confidence,
                        duration=duration)
            self.record_success()

            return AgentResult(
                success=True,
                output=output,
                metadata={
                    "query":       search_query,
                    "num_results": num_results,
                    "sources":     successful_sources,
                },
                confidence=confidence,
                agent_name=self.name,
                duration_sec=duration,
            )

        except Exception as e:
            logger.error("researcher_error", error=str(e))
            self.record_failure()
            return AgentResult(
                success=False,
                output="",
                errors=[str(e)],
                confidence=0.0,
                agent_name=self.name,
                duration_sec=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_search_query(self, task: str) -> str:
        """Strip action-verb prefixes and return the bare search subject."""
        action_phrases = [
            "use web_search to find", "use web_search to search for",
            "use semantic_scholar_search to find",
            "use wikipedia_search to find",
            "use web_search to", "search for", "research", "find",
            "look up", "investigate", "explore", "discover",
            "learn about", "information about", "information on",
        ]
        query = task
        for phrase in action_phrases:
            query = re.sub(re.escape(phrase), " ", query, flags=re.IGNORECASE).strip()

        filler = [r"\bthe\b", r"\babout\b", r"\bregarding\b", r"\bconcerning\b"]
        for pattern in filler:
            query = re.sub(pattern, " ", query, flags=re.IGNORECASE)

        query = re.sub(r"\s+", " ", query).strip()
        return query if len(query) >= 5 else task

    def _extract_first_url(self, search_output: str) -> Optional[str]:
        """Pull the first https URL from formatted search output."""
        match = re.search(r"URL:\s*(https?://\S+)", search_output)
        return match.group(1).strip() if match else None

    def _deduplicate(self, text: str) -> str:
        """Remove result blocks whose first 80 chars duplicate an earlier block."""
        seen: set = set()
        blocks = re.split(r"\n(?=\d+\.)", text)
        unique: List[str] = []
        for block in blocks:
            key = block[:80].strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(block)
        return "\n".join(unique)

    def _format_research_output(
        self,
        query: str,
        search_output: str,
        sources: List[str],
        num_results: int,
    ) -> str:
        out  = "RESEARCH FINDINGS\n"
        out += f"Query:   {query}\n"
        out += f"Sources: {', '.join(sources)}\n"
        out += f"Results: {num_results}\n"
        out += f"\n{'-' * 60}\n\n"
        out += search_output
        return out
