import json
import structlog
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.utils.llm import call_llm

logger = structlog.get_logger()


class ReasoningOutput(BaseModel):
    """Structured reasoning output"""
    problem_type: str  # e.g., "file_operation", "web_research", "calculation"
    strategy: str  # High-level approach
    needs_memory: bool  # Should we check past experiences?
    needs_search: bool  # Should we search the web?
    likely_tools: List[str]  # Expected tools to use
    uncertainties: List[str]  # What could go wrong?
    confidence: float  # 0.0 to 1.0


class ReasonerAgent:
    """
    Pre-planning reasoning agent.
    
    RUNS BEFORE: Planner
    PURPOSE: Strategic thinking before action
    MODEL: Claude Haiku (fast + smart for analysis)
    """
    
    SYSTEM_PROMPT = """You are a strategic reasoning agent. Your job is to analyze a task BEFORE planning begins.

Your analysis helps the system:
1. Choose the right approach
2. Avoid known failure patterns
3. Use resources efficiently (memory, search, tools)
4. Set realistic expectations

ANALYSIS FRAMEWORK:

Problem Types:
- file_operation: Reading, writing, managing files
- web_research: Searching for information online
- calculation: Math, data processing, algorithms
- data_transformation: Parse, convert, format data
- system_operation: Shell commands, system queries
- mixed: Combination of above

Strategy Guidelines:
- For file operations: Check if file exists first, use file_* tools
- For web research: Use web_search, may need multiple searches
- For calculations: Use python_executor with clear logic
- For unknown topics: Definitely need web search
- For repeated tasks: Check memory for past solutions

Tool Predictions:
- file_read, file_write, file_list, file_delete: File operations
- web_search, web_fetch: Internet research
- python_executor: Calculations, data processing
- shell_executor: System commands

Confidence Assessment:
- 0.9-1.0: Very clear task, standard approach
- 0.7-0.9: Clear task, minor uncertainties
- 0.5-0.7: Some ambiguity, multiple approaches possible
- 0.3-0.5: Significant uncertainty, need experimentation
- 0.0-0.3: Very unclear, high risk of failure

RESPONSE FORMAT (JSON only):
{
  "problem_type": "file_operation|web_research|calculation|data_transformation|system_operation|mixed",
  "strategy": "High-level approach in 1-2 sentences",
  "needs_memory": true|false,
  "needs_search": true|false,
  "likely_tools": ["tool1", "tool2"],
  "uncertainties": ["uncertainty1", "uncertainty2"],
  "confidence": 0.85
}

EXAMPLES:

Task: "Create a file called test.txt with hello world"
{
  "problem_type": "file_operation",
  "strategy": "Use file_write to create new file with specified content",
  "needs_memory": false,
  "needs_search": false,
  "likely_tools": ["file_write"],
  "uncertainties": ["file may already exist"],
  "confidence": 0.95
}

Task: "Search for the latest developments in quantum computing"
{
  "problem_type": "web_research",
  "strategy": "Use web_search to find recent articles, may need multiple searches for depth",
  "needs_memory": false,
  "needs_search": true,
  "likely_tools": ["web_search", "web_fetch"],
  "uncertainties": ["topic is rapidly evolving", "need to verify recency of sources"],
  "confidence": 0.75
}

Task: "Calculate fibonacci(100)"
{
  "problem_type": "calculation",
  "strategy": "Use python_executor with iterative approach to avoid recursion limits",
  "needs_memory": true,
  "needs_search": false,
  "likely_tools": ["python_executor"],
  "uncertainties": ["large number may need special handling"],
  "confidence": 0.9
}

RESPOND ONLY WITH JSON."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
    
    async def reason(
        self,
        task_description: str,
        past_memories: Optional[List[Dict]] = None
    ) -> ReasoningOutput:
        """
        Analyze task and produce reasoning output
        
        Args:
            task_description: The user's task
            past_memories: Optional context from similar past tasks
            
        Returns:
            ReasoningOutput with strategic analysis
        """
        logger.info("reasoner_starting", task=task_description)
        
        # Build context
        memory_context = ""
        if past_memories:
            memory_context = f"\n\nPAST SIMILAR TASKS:\n{json.dumps(past_memories, indent=2)}"
        
        user_prompt = f"""TASK TO ANALYZE:
{task_description}
{memory_context}

Analyze this task strategically. Consider:
- What type of problem is this?
- What approach would work best?
- Do we need to check past experiences (memory)?
- Do we need to search the web?
- What tools will likely be needed?
- What could go wrong?
- How confident are you in this assessment?

Return JSON only."""
        
        try:
            response = await call_llm(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.2,  # Some creativity in analysis
            )
            
            # Parse response
            response_text = response.strip()
            
            # Handle markdown code blocks
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1] if "\n" in response_text else response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
            
            # Extract JSON
            start = response_text.find("{")
            end = response_text.rfind("}")
            
            if start != -1 and end != -1 and end > start:
                response_text = response_text[start:end+1]
            
            reasoning_data = json.loads(response_text)
            
            # Validate and create output
            reasoning = ReasoningOutput(
                problem_type=reasoning_data.get("problem_type", "mixed"),
                strategy=reasoning_data.get("strategy", ""),
                needs_memory=reasoning_data.get("needs_memory", False),
                needs_search=reasoning_data.get("needs_search", False),
                likely_tools=reasoning_data.get("likely_tools", []),
                uncertainties=reasoning_data.get("uncertainties", []),
                confidence=float(reasoning_data.get("confidence", 0.5))
            )
            
            logger.info(
                "reasoner_completed",
                problem_type=reasoning.problem_type,
                confidence=reasoning.confidence,
                needs_search=reasoning.needs_search,
                needs_memory=reasoning.needs_memory
            )
            
            return reasoning
        
        except json.JSONDecodeError as e:
            logger.error("reasoner_json_error", error=str(e), response=response_text[:500])
            # Return conservative default
            return ReasoningOutput(
                problem_type="mixed",
                strategy="Unable to analyze - proceeding with caution",
                needs_memory=True,
                needs_search=True,
                likely_tools=[],
                uncertainties=["analysis failed"],
                confidence=0.3
            )
        
        except Exception as e:
            logger.error("reasoner_error", error=str(e))
            # Return conservative default
            return ReasoningOutput(
                problem_type="mixed",
                strategy="Error in analysis - using safe defaults",
                needs_memory=True,
                needs_search=True,
                likely_tools=[],
                uncertainties=["analysis error"],
                confidence=0.2
            )
    
    def should_use_memory(self, reasoning: ReasoningOutput) -> bool:
        """Decide if memory lookup is worth the cost"""
        return reasoning.needs_memory or reasoning.confidence < 0.7
    
    def should_use_search(self, reasoning: ReasoningOutput) -> bool:
        """Decide if web search is needed"""
        return reasoning.needs_search