import structlog
from typing import List, Dict, Optional, Tuple
from app.agents.reasoner import ReasoningOutput

logger = structlog.get_logger()


class SearchDecider:
    """
    Intelligent search decision logic.

    PRINCIPLE: Only search when we actually need external information.
    """

    CONFIDENCE_THRESHOLD = 0.6

    SEARCH_INDICATORS = [
        "latest", "recent", "current", "today", "news",
        "what is", "who is", "where is", "when did",
        "search for", "find", "look up", "research",
        "discover", "investigate", "explore",
    ]

    NO_SEARCH_INDICATORS = [
        "create file", "read file", "write file", "delete file",
        "list files", "calculate", "compute", "parse",
        "format", "convert", "sort", "filter",
    ]

    def should_search(
        self,
        task_description: str,
        reasoning: ReasoningOutput,
        memory_confidence: Optional[float] = None,
        similar_memories: Optional[List[Dict]] = None,
    ) -> Tuple[bool, str]:
        """
        Decide if web search is needed.

        Returns:
            (should_search: bool, reason: str)
        """
        task_lower = task_description.lower()

        if reasoning.needs_search:
            logger.info("search_decision_reasoner", decision=True)
            return True, "Reasoner determined search is needed"

        for indicator in self.NO_SEARCH_INDICATORS:
            if indicator in task_lower:
                logger.info("search_decision_no_keyword_override", indicator=indicator)
                return False, f"Task is internal operation: '{indicator}'"

        for indicator in self.SEARCH_INDICATORS:
            if indicator in task_lower:
                logger.info("search_decision_keyword", indicator=indicator)
                return True, f"Task contains search indicator: '{indicator}'"

        if reasoning.confidence < self.CONFIDENCE_THRESHOLD:
            logger.info("search_decision_low_confidence", confidence=reasoning.confidence)
            return True, f"Low confidence ({reasoning.confidence:.2f}) - searching for clarity"

        if memory_confidence and memory_confidence > 0.8:
            logger.info("search_decision_strong_memory", memory_confidence=memory_confidence)
            return False, f"Strong memory match ({memory_confidence:.2f}) - no search needed"

        if similar_memories:
            avg_success = (
                sum(m.get("success_rate", 0) for m in similar_memories)
                / len(similar_memories)
            )
            if avg_success > 0.8:
                logger.info("search_decision_memory_success", avg_success=avg_success)
                return False, f"Past successes available ({avg_success:.2f}) - no search needed"

        if reasoning.problem_type in ["calculation", "file_operation", "system_operation"]:
            logger.info("search_decision_problem_type", problem_type=reasoning.problem_type)
            return False, f"Problem type '{reasoning.problem_type}' doesn't need web search"

        logger.info("search_decision_default", decision=False)
        return False, "No strong indicators for search - proceeding without"