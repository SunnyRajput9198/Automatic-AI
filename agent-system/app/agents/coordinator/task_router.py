import structlog
from app.agents.memory.agent_performance_memory import AgentPerformanceMemory
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
        "search", "research", "find", "investigate", "explore", "discover",
        "lookup", "query", "latest", "current", "what is", "who is",
        "where is", "when did", "tell me about", "explain", "how does",
        "compare", "analyze", "analyse", "difference between", "overview of",
    ]

    ENGINEER_KEYWORDS = [
        "code", "python", "calculate", "compute", "script", "program",
        "implement", "develop", "build", "create file", "algorithm",
        "function", "class", "run", "execute", "test", "debug", "fix",
        "parse", "convert", "transform", "generate", "automate",
    ]

    WRITER_KEYWORDS = [
        "write", "draft", "compose", "document", "article", "blog",
        "post", "summary", "report", "format", "create content",
        "essay", "email", "letter", "outline", "template", "describe",
    ]

    def __init__(self):
        self.pref_memory = AgentPreferenceMemory()
        self.performance_memory = AgentPerformanceMemory()

    async def route(self, task: str) -> RoutingDecision:
        """
        Route task to appropriate agents.

        Args:
            task: Task description

        Returns:
            RoutingDecision with agents and execution mode
        """
        task_lower = task.lower()
        preferred_agent = self.pref_memory.get_preferred_agent(task)

        # Check performance memory — only use it when the top-performing agent's
        # role actually matches the type of task being routed.
        # Avoids locking ALL tasks to the first agent that hits 80% success.
        best_agent_for_task = None
        best_rate = 0.0
        task_lower_check = task.lower()
        for agent_name, stats in self.performance_memory.all().items():
            role = stats.get("role", "")
            success_rate = stats.get("success_rate", 0)
            if success_rate <= best_rate or success_rate < 0.8:
                continue
            # Only promote this agent if its role matches the task's signal
            role_fits = (
                (role == "researcher" and any(k in task_lower_check for k in self.RESEARCHER_KEYWORDS))
                or (role == "engineer"  and any(k in task_lower_check for k in self.ENGINEER_KEYWORDS))
                or (role == "writer"    and any(k in task_lower_check for k in self.WRITER_KEYWORDS))
            )
            if role_fits:
                best_rate = success_rate
                best_agent_for_task = role

        if best_agent_for_task:
            logger.info(
                "router_using_performance_memory",
                agent=best_agent_for_task,
                success_rate=best_rate,
            )
            return RoutingDecision(
                agents_needed=[best_agent_for_task],
                execution_mode="sequential",
                reasoning=f"Performance memory selected {best_agent_for_task} (success_rate={best_rate:.2f})",
                confidence=min(best_rate, 0.95),
            )

        if preferred_agent:
            logger.info("router_using_agent_preference", agent=preferred_agent)
            return RoutingDecision(
                agents_needed=[preferred_agent],
                execution_mode="sequential",
                reasoning=f"Agent preference memory selected {preferred_agent}",
                confidence=0.95,
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
        # No keyword match — ask Claude
        if not agents_needed:
            logger.info("router_no_keyword_match_using_reasoner", task=task)
            from app.agents.reasoner import ReasonerAgent

            reasoner = ReasonerAgent()
            reasoning = await reasoner.reason(task)

            # map problem_type to agent
            type_to_agent = {
                "file_operation":      "engineer",
                "web_research":        "researcher",
                "calculation":         "engineer",
                "data_transformation": "engineer",
                "system_operation":    "engineer",
                "mixed":               "researcher",  # handled below as multi-agent
            }
            if reasoning.problem_type == "mixed":
                agents_needed = ["researcher", "engineer"]
                keywords_found.append("reasoner:mixed")
            else:
                agent = type_to_agent.get(reasoning.problem_type, "researcher")
                agents_needed.append(agent)
                keywords_found.append(f"reasoner:{reasoning.problem_type}")

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
            confidence=confidence,
        )

        logger.info(
            "task_routed",
            agents=agents_needed,
            mode=execution_mode,
            confidence=confidence,
        )

        return decision

    def _determine_execution_mode(
        self, task_lower: str, agents_needed: List[str]
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
        # Check for sequential indicators (must be standalone time-order words)
        sequential_words = ["then", "after that", "once done", "first then", "followed by"]
        for phrase in sequential_words:
            if phrase in task_lower:
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
        self, task_lower: str, agents_needed: List[str], keywords_found: List[str]
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
