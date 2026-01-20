import structlog
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.agents.base_agent import BaseAgent, AgentResult, BaseAgent
from app.agents.coordinator.task_router import TaskRouter

logger = structlog.get_logger()


class CoordinationResult(BaseModel):
    """Result from coordinating multiple agents"""
    success: bool
    final_output: str
    agent_results: List[Dict[str, Any]]  # Results from each agent
    execution_mode: str  # parallel or sequential
    total_agents: int
    successful_agents: int
    failed_agents: int
    reasoning: str


class CoordinatorAgent:
    """
    Coordinates multiple specialist agents.
    
    WORKFLOW:
    1. Analyze task
    2. Route to agents
    3. Execute agents (parallel or sequential)
    4. Aggregate results
    5. Return coordination result
    
    DAY 1 VERSION:
    - Basic routing
    - Sequential execution only
    - Simple result aggregation
    
    Later days will add:
    - Parallel execution
    - Multi-plan generation
    - Self-healing recovery
    """
    
    def __init__(self, available_agents: Dict[str, BaseAgent]):
        """
        Initialize coordinator with available specialist agents.
        
        Args:
            available_agents: Dict mapping role -> agent instance
                             e.g., {"researcher": ResearcherAgent(), ...}
        """
        self.available_agents = available_agents
        self.router = TaskRouter()
        
        logger.info(
            "coordinator_initialized",
            available_agents=list(available_agents.keys())
        )
    
    async def coordinate(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CoordinationResult:
        """
        Coordinate specialist agents to complete the task.
        
        Args:
            task: Task description
            context: Optional context from orchestrator
            
        Returns:
            CoordinationResult with aggregated output
        """
        logger.info("coordinator_starting", task=task)
        
        context = context or {}
        
        # STEP 1: Route task to agents
        routing = self.router.route(task)
        
        logger.info(
            "coordinator_routed",
            agents=routing.agents_needed,
            mode=routing.execution_mode,
            confidence=routing.confidence
        )
        
        # STEP 2: Execute agents (sequential for Day 1)
        agent_results = []
        execution_context = context.copy()
        
        for agent_role in routing.agents_needed:
            # Check if agent is available
            if agent_role not in self.available_agents:
                logger.error(
                    "coordinator_agent_missing",
                    role=agent_role,
                    available=list(self.available_agents.keys())
                )
                
                # Create failure result
                agent_results.append({
                    "agent": agent_role,
                    "success": False,
                    "output": "",
                    "error": f"Agent '{agent_role}' not available"
                })
                continue
            
            # Get agent
            agent = self.available_agents[agent_role]
            
            logger.info("coordinator_executing_agent", agent=agent_role)
            
            # Execute agent
            try:
                result = await agent.execute(task, execution_context)
                
                # Record result
                agent_results.append({
                    "agent": agent.name,
                    "role": agent.role,
                    "success": result.success,
                    "output": result.output,
                    "confidence": result.confidence,
                    "metadata": result.metadata,
                    "errors": result.errors
                })
                
                # Update context for next agent (sequential passing)
                execution_context[f"{agent_role}_output"] = result.output
                execution_context[f"{agent_role}_success"] = result.success
                
                logger.info(
                    "coordinator_agent_completed",
                    agent=agent_role,
                    success=result.success,
                    confidence=result.confidence
                )
                
            except Exception as e:
                logger.error(
                    "coordinator_agent_error",
                    agent=agent_role,
                    error=str(e)
                )
                
                agent_results.append({
                    "agent": agent_role,
                    "success": False,
                    "output": "",
                    "error": str(e)
                })
        
        # STEP 3: Aggregate results
        coordination_result = self._aggregate_results(
            task=task,
            agent_results=agent_results,
            execution_mode=routing.execution_mode,
            routing_reasoning=routing.reasoning
        )
        
        logger.info(
            "coordinator_completed",
            success=coordination_result.success,
            agents_used=coordination_result.total_agents,
            successful=coordination_result.successful_agents
        )
        
        return coordination_result
    
    def _aggregate_results(
        self,
        task: str,
        agent_results: List[Dict[str, Any]],
        execution_mode: str,
        routing_reasoning: str
    ) -> CoordinationResult:
        """
        Aggregate results from multiple agents into final output.
        
        Args:
            task: Original task
            agent_results: List of results from each agent
            execution_mode: parallel or sequential
            routing_reasoning: Why these agents were chosen
            
        Returns:
            CoordinationResult with final output
        """
        successful = [r for r in agent_results if r.get("success", False)]
        failed = [r for r in agent_results if not r.get("success", False)]
        
        # Build final output
        if not successful:
            # All agents failed
            final_output = "All agents failed to complete the task."
            success = False
        else:
            # Combine successful outputs
            outputs = [r["output"] for r in successful if r.get("output")]
            
            if len(outputs) == 1:
                final_output = outputs[0]
            else:
                # Multiple outputs - combine them
                final_output = "\n\n".join([
                    f"=== {r['role'].upper()} OUTPUT ===\n{r['output']}"
                    for r in successful
                ])
            
            success = True
        
        # Build reasoning
        reasoning = f"Routing: {routing_reasoning}. "
        reasoning += f"Executed {len(agent_results)} agents in {execution_mode} mode. "
        reasoning += f"{len(successful)} succeeded, {len(failed)} failed."
        
        return CoordinationResult(
            success=success,
            final_output=final_output,
            agent_results=agent_results,
            execution_mode=execution_mode,
            total_agents=len(agent_results),
            successful_agents=len(successful),
            failed_agents=len(failed),
            reasoning=reasoning
        )