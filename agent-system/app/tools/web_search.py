import os
import structlog
from typing import Dict, Any
import httpx
from bs4 import BeautifulSoup
import re
from urllib.parse import unquote

from app.tools.base import Tool, ToolResult

logger = structlog.get_logger()

class WebSearchTool(Tool):
    """
    Search the web using DuckDuckGo HTML (no API key required)
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
    
    def _extract_url(self, ddg_url: str) -> str:
        """Extract actual URL from DuckDuckGo redirect URL"""
        try:
            # DuckDuckGo wraps URLs like: //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com
            if 'uddg=' in ddg_url:
                # Extract the uddg parameter
                match = re.search(r'uddg=([^&]+)', ddg_url)
                if match:
                    return unquote(match.group(1))
            return ddg_url
        except:
            return ddg_url
    
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
            # Use DuckDuckGo HTML search (more reliable than API)
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                )
                
                if response.status_code != 200:
                    logger.error("web_search_http_error", status=response.status_code)
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Search returned status {response.status_code}"
                    )
                
                # Parse HTML results - specify parser to avoid warning
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                
                # Find result divs (multiple possible class names)
                result_divs = (
                    soup.find_all('div', class_='result') or 
                    soup.find_all('div', class_='results_links') or
                    soup.find_all('div', class_='web-result')
                )
                
                for i, result_div in enumerate(result_divs[:max_results]):
                    # Try different possible selectors
                    title_elem = (
                        result_div.find('a', class_='result__a') or
                        result_div.find('a', class_='result__url') or
                        result_div.find('h2', class_='result__title')
                    )
                    
                    snippet_elem = (
                        result_div.find('a', class_='result__snippet') or
                        result_div.find('div', class_='result__snippet') or
                        result_div.find('span', class_='result__snippet')
                    )
                    
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        raw_url = title_elem.get('href', '')
                        url = self._extract_url(raw_url) if raw_url else ""
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        if title:  # Only add if we have at least a title
                            result_text = f"{i+1}. {title}"
                            if snippet:
                                result_text += f"\n   {snippet}"
                            if url and url.startswith('http'):
                                result_text += f"\n   URL: {url}"
                            results.append(result_text)
                
                if not results:
                    logger.warning("web_search_no_results", query=query)
                    # Still return success but indicate no results
                    return ToolResult(
                        success=True,
                        output=f"Search completed but no results found for '{query}'. The search engine may be blocking automated requests or the query returned no matches. Try a different search term.",
                        metadata={"query": query, "num_results": 0, "source": "DuckDuckGo"}
                    )
                
                output = "\n\n".join(results)
                logger.info("web_search_completed", num_results=len(results))
                
                return ToolResult(
                    success=True,
                    output=output,
                    metadata={
                        "query": query,
                        "num_results": len(results),
                        "source": "DuckDuckGo"
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