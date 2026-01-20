import structlog
from typing import List, Dict, Any
from pydantic import BaseModel
from app.agents.memory.agent_preference_memory import AgentPreferenceMemory
logger = structlog.get_logger()


class RoutingDecision(BaseModel):
    """Routing decision for a task"""
    agents_needed: List[str]  # List of agent roles needed
    execution_mode: str  # "sequential" or "parallel"
    reasoning: str  # Why these agents were chosen
    confidence: float  # Confidence in this routing


class TaskRouter:
    """
    Routes tasks to appropriate specialist agents.
    
    ROUTING RULES:
    - "search", "research", "find" → researcher
    - "code", "calculate", "python", "script" → engineer
    - "write", "draft", "document", "article" → writer
    - Multiple keywords → multiple agents
    
    EXECUTION MODES:
    - parallel: agents can work independently (research + code)
    - sequential: agents depend on each other (research → write)
    """
    
    # Keyword mappings
    RESEARCHER_KEYWORDS = [
        "search", "research", "find", "investigate", "explore",
        "discover", "lookup", "query", "latest", "current"
    ]
    
    ENGINEER_KEYWORDS = [
        "code", "python", "calculate", "compute", "script",
        "program", "implement", "develop", "build", "create file",
        "algorithm", "function", "class"
    ]
    
    WRITER_KEYWORDS = [
        "write", "draft", "compose", "document", "article",
        "blog", "post", "summary", "report", "format"
    ]
    
    def __init__(self):
        self.pref_memory = AgentPreferenceMemory()

    
    def route(self, task: str) -> RoutingDecision:
        """
        Route task to appropriate agents.
        
        Args:
            task: Task description
            
        Returns:
            RoutingDecision with agents and execution mode
        """
        task_lower = task.lower()
        preferred_agent = self.pref_memory.get_preferred_agent(task)

        if preferred_agent:
            logger.info(
                "router_using_agent_preference",
                agent=preferred_agent
            )

            return RoutingDecision(
                agents_needed=[preferred_agent],
                execution_mode="sequential",
                reasoning=f"Agent preference memory selected {preferred_agent}",
                confidence=0.95
            )

        # Detect which agents are needed
        agents_needed = []
        keywords_found = []
        
        # Check for researcher
        for keyword in self.RESEARCHER_KEYWORDS:
            if keyword in task_lower:
                if "researcher" not in agents_needed:
                    agents_needed.append("researcher")
                    keywords_found.append(f"researcher:{keyword}")
                break
        
        # Check for engineer
        for keyword in self.ENGINEER_KEYWORDS:
            if keyword in task_lower:
                if "engineer" not in agents_needed:
                    agents_needed.append("engineer")
                    keywords_found.append(f"engineer:{keyword}")
                break
        
        # Check for writer
        for keyword in self.WRITER_KEYWORDS:
            if keyword in task_lower:
                if "writer" not in agents_needed:
                    agents_needed.append("writer")
                    keywords_found.append(f"writer:{keyword}")
                break
        
        # Default to engineer if no matches
        if not agents_needed:
            agents_needed.append("engineer")
            keywords_found.append("default:engineer")
        
        # Determine execution mode
        execution_mode = self._determine_execution_mode(task_lower, agents_needed)
        
        # Build reasoning
        reasoning = self._build_reasoning(task_lower, agents_needed, keywords_found)
        
        # Calculate confidence
        confidence = self._calculate_confidence(keywords_found)
        
        decision = RoutingDecision(
            agents_needed=agents_needed,
            execution_mode=execution_mode,
            reasoning=reasoning,
            confidence=confidence
        )
        
        logger.info(
            "task_routed",
            agents=agents_needed,
            mode=execution_mode,
            confidence=confidence
        )
        
        return decision
    
    def _determine_execution_mode(
        self,
        task_lower: str,
        agents_needed: List[str]
    ) -> str:
        """
        Decide if agents should run in parallel or sequential.
        
        Sequential indicators:
        - "then", "after", "once", "first...then"
        - researcher + writer (research → write)
        
        Parallel indicators:
        - "and" without temporal words
        - Independent tasks
        """
        # Check for sequential indicators
        sequential_words = ["then", "after", "once", "first", "before"]
        for word in sequential_words:
            if word in task_lower:
                return "sequential"
        
        # If researcher + writer, usually sequential (research first)
        if "researcher" in agents_needed and "writer" in agents_needed:
            return "sequential"
        
        # Multiple agents without sequential words = parallel
        if len(agents_needed) > 1:
            return "parallel"
        
        # Single agent = sequential (doesn't matter)
        return "sequential"
    
    def _build_reasoning(
        self,
        task_lower: str,
        agents_needed: List[str],
        keywords_found: List[str]
    ) -> str:
        """Generate human-readable reasoning for routing decision"""
        if len(agents_needed) == 1:
            return f"Task requires {agents_needed[0]} based on keywords: {', '.join(keywords_found)}"
        else:
            return f"Task requires multiple agents ({', '.join(agents_needed)}) based on keywords: {', '.join(keywords_found)}"
    
    def _calculate_confidence(self, keywords_found: List[str]) -> float:
        """
        Calculate confidence in routing decision.
        
        High confidence: Clear keyword matches
        Low confidence: Default routing
        """
        if any("default" in kw for kw in keywords_found):
            return 0.5  # Low confidence - default routing
        
        if len(keywords_found) >= 2:
            return 0.9  # High confidence - multiple clear signals
        
        return 0.75  # Medium confidence - one clear match