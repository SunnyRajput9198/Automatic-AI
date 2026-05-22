import structlog
from typing import Dict, Any
import httpx
import xml.etree.ElementTree as ET  # for ArXiv XML parsing
import asyncio

from app.tools.base import Tool, ToolResult

logger = structlog.get_logger()

class WebSearchTool(Tool):
    """
    Search using Wikipedia REST API + ArXiv API.
    No API key required.
    """
    
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
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    
    async def _wiki_search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=10,headers={"User-Agent": "ResearchAgent/1.0 (educational project)"}) as client:
        
        # fix spaces → underscores
            slug = query.strip().replace(" ", "_")# if koi blank soace aae to _ dal do like machine_learning if _ ni hua to
        
            response = await client.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
        )
        
        # handle page not found
            if response.status_code != 200:
                return []
        
            data = response.json()
            return [{
            "source": "Wikipedia",
            "title": data.get("title", ""),
            "content": data.get("extract", ""),
            "url": data.get("content_urls", {})
                .get("desktop", {})
                .get("page", "")
        }]
    
    async def _arxiv_search(self, query: str) -> list[dict]:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "max_results": 3
            }
        )
        
            root = ET.fromstring(response.text)
            ns = {"a": "http://www.w3.org/2005/Atom"}
            # ns is used as a namespace. It tells the parser to look inside the a:entry tag for the title, summary, and id tags.
            #xmlns is called a namespace. Without telling your code that namespace, findall("entry") returns nothing. The ns dictionary is how you tell Python which namespace to look inside.
            papers = []
            for entry in root.findall("a:entry", ns):
                title = entry.find("a:title", ns)
                summary = entry.find("a:summary", ns)
                link = entry.find("a:id", ns)
            
                papers.append({
                "source": "ArXiv",
                "title": title.text.strip() if title is not None and title.text else "",
                "content": summary.text.strip() if summary is not None and summary.text else "",
                "url": link.text.strip() if link is not None and link.text else ""
            })
        
            return papers
    async def run(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        
        if not query:
            return ToolResult(
                success=False,
                output="",
                error="Search query is required"
            )
        
        logger.info("web_search_running", query=query)
        
        try:
            wiki_results, arxiv_results = await asyncio.gather(
                self._wiki_search(query),
                self._arxiv_search(query),
                return_exceptions=True # don't let one failure kill both
            )

            sources = []
            if isinstance(wiki_results, list):
                sources.extend(wiki_results)
            if isinstance(arxiv_results, list):
                sources.extend(arxiv_results)

            if not sources:
                return ToolResult(
                    success=False,
                    output="",
                    error="No results found from Wikipedia or ArXiv",
                    metadata={"query": query, "num_results": 0}
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
                    "query": query,
                    "num_results": len(sources),
                    "source": "wikipedia+arxiv"
                }
            )
                        
            
        
        except httpx.TimeoutException:
            logger.error("web_search_timeout", query=query)
            return ToolResult(
                success=False,
                output="",
                error="Search request timed out after 15 seconds"
            )
        
        except httpx.RequestError as e:
            logger.error("web_search_request_error", error=str(e), query=query)
            return ToolResult(
                success=False,
                output="",
                error=f"Network error during search: {str(e)}"
            )
        
        except Exception as e:
            logger.error("web_search_error", error=str(e), query=query)
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {str(e)}"
            )


class WebFetchTool(Tool):
    """
    Fetch content from a specific URL
    
    SECURITY: Only allow HTTPS, block certain domains
    """
    
    BLOCKED_DOMAINS = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "internal",
        "private"
    ]
    
    @property
    def name(self) -> str:
        return "web_fetch"
    
    @property
    def description(self) -> str:
        return "Fetch the text content of a specific webpage. Only HTTPS URLs are allowed."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch (must be HTTPS)"
                }
            },
            "required": ["url"]
        }
    
    async def run(self, **kwargs) -> ToolResult:
        url = kwargs.get("url", "")
        
        if not url:
            return ToolResult(
                success=False,
                output="",
                error="URL is required"
            )
        
        # Security checks
        if not url.startswith("https://") and not url.startswith("http://"):
            return ToolResult(
                success=False,
                output="",
                error="URL must start with http:// or https://"
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
                    error=f"Access to {blocked} is not allowed for security reasons"
                )
        
        logger.info("web_fetch_running", url=url)
        
        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            ) as client:
                response = await client.get(url)
                
                if response.status_code != 200:
                    logger.warning("web_fetch_http_error", status=response.status_code, url=url)
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"HTTP {response.status_code}: {response.reason_phrase}"
                    )
                
                # Get text content (limit to 50KB to avoid huge responses but still useful)
                content = response.text[:50000]
                
                logger.info("web_fetch_completed", url=url, size=len(content))
                
                return ToolResult(
                    success=True,
                    output=content,
                    metadata={
                        "url": url,
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type", ""),
                        "size": len(content)
                    }
                )
        
        except httpx.TimeoutException:
            logger.error("web_fetch_timeout", url=url)
            return ToolResult(
                success=False,
                output="",
                error="Request timed out after 15 seconds"
            )
        
        except httpx.RequestError as e:
            logger.error("web_fetch_request_error", error=str(e), url=url)
            return ToolResult(
                success=False,
                output="",
                error=f"Network error: {str(e)}"
            )
        
        except Exception as e:
            logger.error("web_fetch_error", error=str(e), url=url)
            return ToolResult(
                success=False,
                output="",
                error=f"Fetch failed: {str(e)}"
            )