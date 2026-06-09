import re
import structlog
from typing import Dict, Any
import httpx
import xml.etree.ElementTree as ET  # for ArXiv XML parsing
import asyncio

from app.tools.base import Tool, ToolResult

logger = structlog.get_logger()

class WebSearchTool(Tool):
    """
    Multi-source web search tool for research and technical queries.

    Sources:
    - Wikipedia for general knowledge
    - ArXiv for research papers
    - PubMed for scientific literature
    - GitHub for repositories and code examples

    Returns normalized search results from multiple sources.
    """

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=10,
            headers={"User-Agent": "ResearchAgent/1.0 (educational project)"},
        )

    async def close(self):
        await self.client.aclose()

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information. Returns top results with titles and snippets."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def _enhance_wiki_query(self, query: str) -> str:
        tech_terms = {
            "docker": "Docker software containerization",
            "python": "Python programming language",
            "rust": "Rust programming language",
            "go": "Go programming language",
            "swift": "Swift programming language Apple",
            "ruby": "Ruby programming language",
            "flask": "Flask Python web framework",
            "django": "Django Python web framework",
            "react": "React JavaScript library",
            "vue": "Vue.js JavaScript framework",
            "angular": "Angular JavaScript framework",
            "spark": "Apache Spark big data",
            "kafka": "Apache Kafka messaging",
            "redis": "Redis database cache",
            "nginx": "Nginx web server",
            "kubernetes": "Kubernetes container orchestration",
            "git": "Git version control software",
        }

        query_lower = query.lower().strip()

        for term, enhanced in tech_terms.items():
            if re.search(rf"\b{re.escape(term)}\b", query_lower):
                return enhanced

        return query

    def _is_technical_query(self, query: str) -> bool:
        """
        Returns True if query is technical/research — use all sources.
        Returns False if general knowledge — use only Wikipedia.
        """
        general_indicators = [
            "what is",
            "what are",
            "who is",
            "who are",
            "explain",
            "tell me about",
            "how does",
            "why is",
            "difference between",
            "compare",
            "vs",
            "versus",
            "define",
            "definition",
            "meaning of",
            "introduction to",
        ]

        technical_indicators = [
            "research",
            "paper",
            "algorithm",
            "implementation",
            "architecture",
            "framework",
            "library",
            "model",
            "dataset",
            "benchmark",
            "sota",
            "state of the art",
            "arxiv",
            "pubmed",
            "github",
            "code",
            "repository",
        ]

        query_lower = query.lower()

        # If explicitly technical — use all sources
        for indicator in technical_indicators:
            if indicator in query_lower:
                return True

        # If general question — Wikipedia only
        for indicator in general_indicators:
            if indicator in query_lower:
                return False

        # Default — use all sources
        return True

    async def _wiki_search(self, query: str) -> list[dict]:

        # Step 1 — search for the right page
        search_resp = await self.client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": self._enhance_wiki_query(query),
                "format": "json",
            },
        )
        if search_resp.status_code != 200:
            return []

        hits = search_resp.json().get("query", {}).get("search", [])
        if not hits:
            return []

        # Step 2 — get summary of top result
        title = hits[0]["title"]
        slug = title.replace(" ", "_")

        response = await self.client.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        )
        if response.status_code != 200:
            return []

        data = response.json()
        return [
            {
                "source": "Wikipedia",
                "title": data.get("title", ""),
                "content": data.get("extract", ""),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            }
        ]

    async def _arxiv_search(self, query: str) -> list[dict]:

        response = await self.client.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": 3},
        )

        root = ET.fromstring(response.text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        # arvix retuns xml, so we need to parse it
        # ns is used as a namespace. It tells the parser to look inside the a:entry tag for the title, summary, and id tags.
        # xmlns is called a namespace. Without telling your code that namespace, findall("entry") returns nothing. The ns dictionary is how you tell Python which namespace to look inside.
        papers = []
        for entry in root.findall("a:entry", ns):
            title = entry.find("a:title", ns)
            summary = entry.find("a:summary", ns)
            link = entry.find("a:id", ns)

            papers.append(
                {
                    "source": "ArXiv",
                    "title": (
                        title.text.strip() if title is not None and title.text else ""
                    ),
                    "content": (
                        summary.text.strip()
                        if summary is not None and summary.text
                        else ""
                    ),
                    "url": (
                        link.text.strip() if link is not None and link.text else ""
                    ),
                }
            )

        return papers

    async def _pubmed_search(self, query: str) -> list[dict]:
        # Step 1 — get IDs
        search_resp = await self.client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": 3, "retmode": "json"},
        )
        if search_resp.status_code != 200:
            return []

        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        # Step 2 — get details
        summary_resp = await self.client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
        )
        if summary_resp.status_code != 200:
            return []

        result = summary_resp.json().get("result", {})
        papers = []
        for uid in ids:
            paper = result.get(uid, {})
            title = paper.get("title", "")
            if not title:
                continue
            papers.append(
                {
                    "source": "PubMed",
                    "title": title,
                    "content": f"Authors: {', '.join(a.get('name','') for a in paper.get('authors', [])[:3])}. Published: {paper.get('pubdate', '')}.",
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                }
            )
        return papers

    async def _github_search(self, query: str) -> list[dict]:
        resp = await self.client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "per_page": 3},
        )
        if resp.status_code != 200:
            return []

        items = resp.json().get("items", [])
        repos = []
        for item in items:
            description = item.get("description") or "No description"
            repos.append(
                {
                    "source": "GitHub",
                    "title": item.get("full_name", ""),
                    "content": f"{description}. Stars: {item.get('stargazers_count', 0)}. Language: {item.get('language', 'Unknown')}.",
                    "url": item.get("html_url", ""),
                }
            )
        return repos

    async def run(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)

        if not query:
            return ToolResult(
                success=False, output="", error="Search query is required"
            )

        logger.info("web_search_running", query=query)

        try:
            is_technical = self._is_technical_query(query)
            logger.info("web_search_query_type", query=query, is_technical=is_technical)

            if is_technical:
                wiki_results, arxiv_results, pubmed_results, github_results = (
                    await asyncio.gather(
                        self._wiki_search(query),
                        self._arxiv_search(query),
                        self._pubmed_search(query),
                        self._github_search(query),
                        return_exceptions=True,
                    )
                )
                results_list = [
                    wiki_results,
                    arxiv_results,
                    pubmed_results,
                    github_results,
                ]
            else:
                wiki_results = await self._wiki_search(query)
                results_list = [wiki_results]

            sources = []
            for result in results_list:
                if isinstance(result, list):
                    sources.extend(result)

            if not sources:
                return ToolResult(
                    success=False,
                    output="",
                    error="No results found from any search source",
                    metadata={"query": query, "num_results": 0},
                )

            # format sources into text
            formatted = []
            for i, s in enumerate(sources[:max_results], 1):
                formatted.append(
                    f"{i}. {s['source']} — {s['title']}\n"
                    f"   {s['content'][:500]}\n"
                    f"   URL: {s['url']}"
                )

            output = "\n\n".join(formatted)
            logger.info("web_search_completed", num_results=len(sources))

            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "tool_name": "web_search",
                    "query": query,
                    "num_results": len(sources),
                    "source": (
                        "wikipedia+arxiv+pubmed+github" if is_technical else "wikipedia"
                    ),
                },
            )

        except httpx.TimeoutException:
            logger.error("web_search_timeout", query=query)
            return ToolResult(
                success=False,
                output="",
                error="Search request timed out after 15 seconds",
            )

        except httpx.RequestError as e:
            logger.error("web_search_request_error", error=str(e), query=query)
            return ToolResult(
                success=False, output="", error=f"Network error during search: {str(e)}"
            )

        except Exception as e:
            logger.error("web_search_error", error=str(e), query=query)
            return ToolResult(
                success=False, output="", error=f"Search failed: {str(e)}"
            )


class WebFetchTool(Tool):
    """
    Fetch content from a specific URL

    SECURITY: Only allow HTTPS, block certain domains
    """

    BLOCKED_DOMAINS = ["localhost", "127.0.0.1", "0.0.0.0", "internal", "private"]

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch the text content of a specific webpage. Only HTTPS URLs are allowed."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch (must be HTTPS)"}
            },
            "required": ["url"],
        }

    async def run(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "")

        if not url:
            return ToolResult(success=False, output="", error="URL is required")

        # Security checks
        if not url.startswith("https://") and not url.startswith("http://"):
            return ToolResult(
                success=False,
                output="",
                error="URL must start with http:// or https://",
            )

        # Warn about HTTP (but allow it for testing)
        if url.startswith("http://") and "httpbin.org" not in url:
            logger.warning("web_fetch_insecure_url", url=url)

        # Check for blocked domains
        for blocked in self.BLOCKED_DOMAINS:
            if blocked in url.lower():
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Access to {blocked} is not allowed for security reasons",
                )

        logger.info("web_fetch_running", url=url)

        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            ) as client:
                response = await client.get(url)

                if response.status_code != 200:
                    logger.warning(
                        "web_fetch_http_error", status=response.status_code, url=url
                    )
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"HTTP {response.status_code}: {response.reason_phrase}",
                    )

                # Get text content (limit to 50KB to avoid huge responses but still useful)
                content = response.text[:50000]

                logger.info("web_fetch_completed", url=url, size=len(content))

                return ToolResult(
                    success=True,
                    output=content,
                    metadata={
                        "tool_name": "web_fetch",
                        "url": url,
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type", ""),
                        "size": len(content),
                    },
                )

        except httpx.TimeoutException:
            logger.error("web_fetch_timeout", url=url)
            return ToolResult(
                success=False, output="", error="Request timed out after 15 seconds"
            )

        except httpx.RequestError as e:
            logger.error("web_fetch_request_error", error=str(e), url=url)
            return ToolResult(
                success=False, output="", error=f"Network error: {str(e)}"
            )

        except Exception as e:
            logger.error("web_fetch_error", error=str(e), url=url)
            return ToolResult(success=False, output="", error=f"Fetch failed: {str(e)}")
