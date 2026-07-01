import structlog
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import asyncio
from app.agents.base_agent import BaseAgent
from app.agents.coordinator.task_router import TaskRouter
from app.agents.memory.agent_performance_memory import AgentPerformanceMemory
from app.utils.llm import call_llm_with_system
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
1. Route task to appropriate agents (keyword match or ReasonerAgent fallback)
2. Execute agents in parallel or sequential mode
3. Aggregate results into final output

EXECUTION MODES:
- parallel: agents run simultaneously via asyncio.gather
- sequential: agents run one after another, passing context forward

AGENTS SUPPORTED:
- researcher: web research via Semantic Scholar + Wikipedia REST + DuckDuckGo
- engineer: Python execution, file operations
- writer: content generation
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
        self.agent_memory = AgentPerformanceMemory()
    
        logger.info(
            "coordinator_initialized",
            available_agents=list(available_agents.keys())
        )
    
    async def coordinate(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CoordinationResult:

        logger.info("coordinator_starting", task=task)
        context = context or {}

        # STEP 1: Route task
        routing = await self.router.route(task)

        logger.info(
            "coordinator_routed",
            agents=routing.agents_needed,
            mode=routing.execution_mode,
            confidence=routing.confidence
        )

        agent_results = []
        execution_context = context.copy()

        # ── Parallel execution ────────────────────────────────────────────
        if routing.execution_mode == "parallel":
            logger.info("coordinator_parallel_execution")

            coroutines = []
            roles = []

            for agent_role in routing.agents_needed:
                agent = self.available_agents.get(agent_role)
                if agent:
                    coroutines.append(agent.execute(task, execution_context))
                    roles.append(agent_role)

            results = await asyncio.gather(*coroutines, return_exceptions=True)

            for role, result in zip(roles, results):
                # Guard against exceptions first
                if isinstance(result, BaseException):
                    agent_results.append({
                        "agent": role,
                        "role": role,
                        "success": False,
                        "output": "",
                        "error": str(result)
                    })
                    continue

                # Safe to access AgentResult attributes here
                agent_results.append({
                    "agent": result.agent_name,
                    "role": role,
                    "success": result.success,
                    "output": result.output,
                    "confidence": result.confidence,
                    "metadata": result.metadata,
                    "errors": result.errors
                })

                # Update memory and context only for non-exception results
                agent = self.available_agents.get(role)
                if agent:
                    self.agent_memory.update(agent.name, agent.get_stats())

                if result.success:
                    execution_context[f"{role}_output"] = result.output
                    execution_context[f"{role}_success"] = True

        else:

            for agent_role in routing.agents_needed:

                if agent_role not in self.available_agents:
                    logger.error(
                        "coordinator_agent_missing",
                        role=agent_role
                    )
                    agent_results.append({
                        "agent": agent_role,
                        "role": agent_role,
                        "success": False,
                        "output": "",
                        "error": "Agent not available"
                    })
                    continue

                agent = self.available_agents[agent_role]

                logger.info("coordinator_executing_agent", agent=agent_role)

                try:
                    result = await agent.execute(task, execution_context)
                    self.agent_memory.update(
                            agent.name,
                            agent.get_stats()
                        )
                    agent_results.append({
                        "agent": agent.name,
                        "role": agent.role,
                        "success": result.success,
                        "output": result.output,
                        "confidence": result.confidence,
                        "metadata": result.metadata,
                        "errors": result.errors
                    })

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
                        "role": agent_role,
                        "success": False,
                        "output": "",
                        "error": str(e)
                    })

        # ==================================================
        # STEP 3: Aggregate
        # ==================================================
        logger.info(
    "coordinator_raw_results",
    results=agent_results
)
        coordination_result = self._aggregate_results(
            task=task,
            agent_results=agent_results,
            execution_mode=routing.execution_mode,
            routing_reasoning=routing.reasoning
        )

        # LLM synthesis pass when multiple agents succeeded
        successful_results = [r for r in agent_results if r.get("success")]
        if len(successful_results) >= 2:
            try:
                synthesized = await self._synthesize_outputs(task, successful_results)
                coordination_result = coordination_result.model_copy(
                    update={"final_output": synthesized}
                )
                logger.info("coordinator_synthesis_applied")
            except Exception as e:
                logger.error("coordinator_synthesis_failed", error=str(e))

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
        successful = [r for r in agent_results if r.get("success", False)]
        failed     = [r for r in agent_results if not r.get("success", False)]

        if not successful:
            final_output = "All agents failed to complete the task."
            success = False
        elif len(successful) == 1:
            # Single agent — use its output directly, no synthesis needed
            final_output = successful[0]["output"]
            success = True
        else:
            # Multiple agents — concatenate with clear headers
            # (LLM synthesis happens asynchronously; for now produce a clean
            #  structured string so the caller always gets something useful)
            final_output = "\n\n".join([
                f"=== {r['role'].upper()} OUTPUT ===\n{r['output']}"
                for r in successful
            ])
            success = True

        reasoning  = f"Routing: {routing_reasoning}. "
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
            reasoning=reasoning,
        )

    async def _synthesize_outputs(
        self, task: str, outputs: List[Dict[str, Any]]
    ) -> str:
        """
        LLM synthesis pass: combine multiple agent outputs into one
        cohesive answer. Only called when ≥2 agents succeeded.
        """
        combined = "\n\n".join([
            f"=== {r['role'].upper()} OUTPUT ===\n{r['output'][:1500]}"
            for r in outputs
        ])
        system_prompt = (
            "You are a result synthesizer. You receive outputs from multiple specialist "
            "agents that worked on the same task. Your job is to combine them into a "
            "single, well-structured, non-redundant answer. "
            "Remove duplicates, reconcile any conflicts, and present the best composite result."
        )
        user_prompt = (
            f"ORIGINAL TASK:\n{task}\n\n"
            f"AGENT OUTPUTS:\n{combined}\n\n"
            "Produce a unified, complete answer."
        )
        try:
            return await call_llm_with_system(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=2000,
            )
        except Exception as e:
            logger.error("coordinator_synthesis_error", error=str(e))
            return combined  # fall back to raw concat