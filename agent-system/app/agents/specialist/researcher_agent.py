import re
import structlog
import time
from typing import Dict, Any, Optional
from app.agents.base_agent import BaseAgent, AgentResult
from app.tools.web_search import WebSearchTool, WebFetchTool

logger = structlog.get_logger()

# Minimum results before we bother fetching a full page
_FETCH_THRESHOLD = 3


class ResearcherAgent(BaseAgent):
    """
    Specialist agent for web research.

    Pipeline:
    1. Extract clean search query from task
    2. Run WebSearchTool (Wikipedia + DuckDuckGo ± ArXiv ± PubMed)
    3. If fewer than _FETCH_THRESHOLD snippets returned, fetch the top URL
       for fuller content
    4. Deduplicate overlapping snippets across sources
    5. Return structured findings
    """

    def __init__(self, name: str = "researcher_001"):
        super().__init__(
            name=name, role="researcher", allowed_tools=["web_search", "web_fetch"]
        )
        self.web_search = WebSearchTool()
        self.web_fetch  = WebFetchTool()

    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        start_time = time.time()
        logger.info("researcher_executing", task=task)

        try:
            search_query = self._extract_search_query(task)
            logger.info("researcher_searching", query=search_query)

            search_result = await self.web_search.run(query=search_query, max_results=6)

            if not search_result.success:
                logger.error("researcher_search_failed", error=search_result.error)
                self.record_failure()
                return AgentResult(
                    success=False,
                    output="",
                    errors=[search_result.error or "Search failed"],
                    confidence=0.0,
                    agent_name=self.name,
                    duration_sec=time.time() - start_time,
                )

            num_results  = search_result.metadata.get("num_results", 0)
            search_output = search_result.output

            # ── Fetch top URL if snippets are thin ───────────────────────
            if num_results < _FETCH_THRESHOLD:
                top_url = self._extract_first_url(search_output)
                if top_url:
                    logger.info("researcher_fetching_url", url=top_url, reason="thin snippets")
                    fetch_result = await self.web_fetch.run(url=top_url)
                    if fetch_result.success and fetch_result.output:
                        # Append truncated page content as an extra source
                        search_output += (
                            f"\n\nFetched full page from {top_url}:\n"
                            f"{fetch_result.output[:1500]}"
                        )
                        num_results += 1

            # ── Deduplicate overlapping snippets ─────────────────────────
            search_output = self._deduplicate(search_output)

            output = self._format_research_output(
                query=search_query,
                search_output=search_output,
                metadata=search_result.metadata,
                num_results=num_results,
            )

            confidence = min(0.95, 0.5 + num_results * 0.1)
            duration   = time.time() - start_time

            logger.info(
                "researcher_completed",
                num_results=num_results,
                confidence=confidence,
                duration=duration,
            )
            self.record_success()

            return AgentResult(
                success=True,
                output=output,
                metadata={
                    "query":       search_query,
                    "num_results": num_results,
                    "source":      search_result.metadata.get("source", "web_search"),
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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_search_query(self, task: str) -> str:
        """Extract a clean search query, stripping action-verb prefixes."""
        action_phrases = [
            "use web_search to find", "use web_search to search for",
            "use web_search to", "search for", "research", "find",
            "look up", "investigate", "explore", "discover", "learn about",
            "information about", "information on",
        ]
        query = task
        for phrase in action_phrases:
            query = re.sub(re.escape(phrase), " ", query, flags=re.IGNORECASE).strip()

        # Only strip common filler words when they appear at word boundaries
        # (avoid stripping "for" from "formula" etc.)
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
        """
        Remove result entries whose first 80 chars duplicate an earlier entry.
        Operates on the numbered-result blocks the search tool produces.
        """
        seen: set = set()
        blocks = re.split(r"\n(?=\d+\.)", text)
        unique_blocks = []
        for block in blocks:
            key = block[:80].strip().lower()
            if key not in seen:
                seen.add(key)
                unique_blocks.append(block)
        return "\n".join(unique_blocks)

    def _format_research_output(
        self,
        query: str,
        search_output: str,
        metadata: Dict[str, Any],
        num_results: int,
    ) -> str:
        output  = "RESEARCH FINDINGS\n"
        output += f"Query: {query}\n"
        output += f"Source: {metadata.get('source', 'web_search')}\n"
        output += f"Results: {num_results}\n"
        output += f"\n{'-'*60}\n\n"
        output += search_output
        return output
