import structlog
import time
from typing import Dict, Any, Optional

from app.agents.base_agent import BaseAgent, AgentResult
from app.tools.web_search import WebSearchTool, WebFetchTool
from dotenv import load_dotenv
load_dotenv()
logger = structlog.get_logger()


class ResearcherAgent(BaseAgent):
    """
    Specialist agent for web research.
    
    Uses web search and fetch tools to gather information.
    Returns structured research results with sources.
    """
    
    def __init__(self, name: str = "researcher_001"):
        super().__init__(
            name=name,
            role="researcher",
            allowed_tools=["web_search", "web_fetch"]
        )
        
        # Initialize tools
        self.web_search = WebSearchTool()
        self.web_fetch = WebFetchTool()
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute research task.
        
        Args:
            task: Research task description
            context: Optional context from coordinator
            
        Returns:
            AgentResult with research findings
        """
        start_time = time.time()
        
        logger.info("researcher_executing", task=task)
        
        try:
            # Extract search query from task
            search_query = self._extract_search_query(task)
            
            logger.info("researcher_searching", query=search_query)
            
            # Perform web search
            search_result = await self.web_search.run(
                query=search_query,
                max_results=5
            )
            
            if not search_result.success:
                logger.error("researcher_search_failed", error=search_result.error)
                self.record_failure()
                
                return AgentResult(
                    success=False,
                    output="",
                    errors=[search_result.error or "Search failed"],
                    confidence=0.0,
                    agent_name=self.name,
                    duration_sec=time.time() - start_time
                )
            
            # Format research output
            output = self._format_research_output(
                query=search_query,
                search_output=search_result.output,
                metadata=search_result.metadata
            )
            
            # Calculate confidence based on results
            num_results = search_result.metadata.get("num_results", 0)
            confidence = min(0.95, 0.5 + (num_results * 0.1))
            
            duration = time.time() - start_time
            
            logger.info(
                "researcher_completed",
                num_results=num_results,
                confidence=confidence,
                duration=duration
            )
            
            self.record_success()
            
            return AgentResult(
                success=True,
                output=output,
                metadata={
                    "query": search_query,
                    "num_results": num_results,
                    "source": search_result.metadata.get("source", "web_search")
                },
                confidence=confidence,
                agent_name=self.name,
                duration_sec=duration
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
                duration_sec=time.time() - start_time
            )
    
    def _extract_search_query(self, task: str) -> str:
        """
        Extract search query from task description.
        
        Removes common action words to get core query.
        """
        # Remove common action words
        query = task.lower()
        
        action_words = [
            "search for", "research", "find", "look up",
            "investigate", "explore", "discover", "learn about"
        ]
        
        for word in action_words:
            query = query.replace(word, "")
        
        # Clean up
        query = query.strip()
        
        # If query is too short, use original task
        if len(query) < 5:
            query = task
        
        return query
    
    def _format_research_output(
        self,
        query: str,
        search_output: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Format research output in a structured way.
        
        Returns:
            Formatted research findings
        """
        output = f"RESEARCH FINDINGS\n"
        output += f"Query: {query}\n"
        output += f"Source: {metadata.get('source', 'web_search')}\n"
        output += f"Results: {metadata.get('num_results', 0)}\n"
        output += f"\n{'-'*60}\n\n"
        output += search_output
        
        return output