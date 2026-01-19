import structlog
from typing import List, Dict, Optional
from app.agents.reasoner import ReasoningOutput

logger = structlog.get_logger()


class SearchDecider:
    """
    Intelligent search decision logic.
    
    PRINCIPLE: Only search when we actually need external information.
    """
    
    # Confidence threshold below which we search
    CONFIDENCE_THRESHOLD = 0.6
    
    # Keywords that almost always indicate need for search
    SEARCH_INDICATORS = [
        "latest", "recent", "current", "today", "news",
        "what is", "who is", "where is", "when did",
        "search for", "find", "look up", "research",
        "discover", "investigate", "explore"
    ]
    
    # Keywords that indicate internal/file operations (no search)
    NO_SEARCH_INDICATORS = [
        "create file", "read file", "write file", "delete file",
        "list files", "calculate", "compute", "parse",
        "format", "convert", "sort", "filter"
    ]
    
    def __init__(self):
        pass
    
    def should_search(
        self,
        task_description: str,
        reasoning: ReasoningOutput,
        memory_confidence: Optional[float] = None,
        similar_memories: Optional[List[Dict]] = None
    ) -> tuple[bool, str]:
        """
        Decide if web search is needed.
        
        Args:
            task_description: The user's task
            reasoning: Output from ReasonerAgent
            memory_confidence: Confidence score from memory retrieval (0-1)
            similar_memories: Past similar tasks from memory
            
        Returns:
            (should_search: bool, reason: str)
        """
        task_lower = task_description.lower()
        
        # Rule 1: Reasoner explicitly says search needed
        if reasoning.needs_search:
            logger.info(
                "search_decision_reasoner",
                decision=True,
                reason="Reasoner determined search is needed"
            )
            return True, "Reasoner determined search is needed"
        
        # Rule 2: Strong search indicators in task
        for indicator in self.SEARCH_INDICATORS:
            if indicator in task_lower:
                logger.info(
                    "search_decision_keyword",
                    decision=True,
                    indicator=indicator
                )
                return True, f"Task contains search indicator: '{indicator}'"
        
        # Rule 3: Strong no-search indicators
        for indicator in self.NO_SEARCH_INDICATORS:
            if indicator in task_lower:
                logger.info(
                    "search_decision_no_keyword",
                    decision=False,
                    indicator=indicator
                )
                return False, f"Task is internal operation: '{indicator}'"
        
        # Rule 4: Low confidence from reasoner
        if reasoning.confidence < self.CONFIDENCE_THRESHOLD:
            logger.info(
                "search_decision_low_confidence",
                decision=True,
                confidence=reasoning.confidence
            )
            return True, f"Low confidence ({reasoning.confidence:.2f}) - searching for clarity"
        
        # Rule 5: Memory has high confidence solution
        if memory_confidence and memory_confidence > 0.8:
            logger.info(
                "search_decision_strong_memory",
                decision=False,
                memory_confidence=memory_confidence
            )
            return False, f"Strong memory match ({memory_confidence:.2f}) - no search needed"
        
        # Rule 6: Similar memories available with good success rate
        if similar_memories:
            avg_success = sum(m.get("success_rate", 0) for m in similar_memories) / len(similar_memories)
            if avg_success > 0.8:
                logger.info(
                    "search_decision_memory_success",
                    decision=False,
                    avg_success=avg_success
                )
                return False, f"Past successes available ({avg_success:.2f}) - no search needed"
        
        # Rule 7: Problem type is calculation or file operation
        if reasoning.problem_type in ["calculation", "file_operation", "system_operation"]:
            logger.info(
                "search_decision_problem_type",
                decision=False,
                problem_type=reasoning.problem_type
            )
            return False, f"Problem type '{reasoning.problem_type}' doesn't need web search"
        
        # Default: Don't search unless explicitly needed
        # (Conservative approach - avoid unnecessary costs)
        logger.info(
            "search_decision_default",
            decision=False,
            reason="No strong indicators for search"
        )
        return False, "No strong indicators for search - proceeding without"
    
    def estimate_search_value(
        self,
        reasoning: ReasoningOutput,
        memory_confidence: Optional[float] = None
    ) -> float:
        """
        Estimate the value of doing a search (0-1 scale).
        Higher = more valuable.
        
        Can be used for cost-benefit analysis.
        """
        value = 0.0
        
        # Search is valuable if:
        # - Reasoner says we need it
        if reasoning.needs_search:
            value += 0.5
        
        # - Confidence is low
        if reasoning.confidence < self.CONFIDENCE_THRESHOLD:
            value += 0.3
        
        # - Memory has low confidence
        if memory_confidence and memory_confidence < 0.5:
            value += 0.2
        
        # - Problem is web research
        if reasoning.problem_type == "web_research":
            value += 0.4
        
        return min(value, 1.0)  # Cap at 1.0