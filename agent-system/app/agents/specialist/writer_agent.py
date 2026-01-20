import json
import structlog
import time
from typing import Dict, Any, Optional

from app.agents.base_agent import BaseAgent, AgentResult
from app.utils.llm import call_llm

logger = structlog.get_logger()


class WriterAgent(BaseAgent):
    """
    Specialist agent for content creation.
    
    Takes information (often from researcher) and creates
    well-formatted, readable content.
    """
    
    SYSTEM_PROMPT = """You are a professional content writer. Your job is to create clear, engaging content.

Given a task and context (often research findings), you:
1. Understand the requirements (article, summary, blog post, etc.)
2. Structure the content appropriately
3. Write in a clear, professional style
4. Format for readability

INPUT:
- Task description (what to write)
- Context (research findings, data, etc.)

OUTPUT FORMAT:
{
  "content": "the written content",
  "title": "content title",
  "word_count": estimated word count,
  "format": "article|blog|summary|report",
  "save_to_file": true|false,
  "filename": "suggested_filename.txt"
}

Make content engaging and well-structured.
Use markdown formatting when appropriate.

RESPOND ONLY WITH JSON."""
    
    def __init__(
        self,
        name: str = "writer_001",
        model: str = "claude-haiku-4-5-20251001"
    ):
        super().__init__(
            name=name,
            role="writer",
            allowed_tools=["file_write"]
        )
        self.model = model
    
    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Execute writing task.
        
        Args:
            task: Writing task description
            context: Optional context (usually from researcher)
            
        Returns:
            AgentResult with written content
        """
        start_time = time.time()
        
        logger.info("writer_executing", task=task)
        
        try:
            # Generate content using LLM
            content_result = await self._generate_content(task, context)
            
            if not content_result:
                self.record_failure()
                return AgentResult(
                    success=False,
                    output="",
                    errors=["Failed to generate content"],
                    confidence=0.0,
                    agent_name=self.name,
                    duration_sec=time.time() - start_time
                )
            
            content = content_result.get("content", "")
            title = content_result.get("title", "Untitled")
            word_count = content_result.get("word_count", 0)
            content_format = content_result.get("format", "text")
            
            # Format output
            output = f"# {title}\n\n"
            output += content
            output += f"\n\n---\n"
            output += f"Format: {content_format} | Words: ~{word_count}"
            
            duration = time.time() - start_time
            
            logger.info(
                "writer_completed",
                title=title,
                word_count=word_count,
                duration=duration
            )
            
            self.record_success()
            
            return AgentResult(
                success=True,
                output=output,
                metadata={
                    "title": title,
                    "word_count": word_count,
                    "format": content_format,
                    "filename": content_result.get("filename")
                },
                confidence=0.8,
                agent_name=self.name,
                duration_sec=duration
            )
        
        except Exception as e:
            logger.error("writer_error", error=str(e))
            self.record_failure()
            
            return AgentResult(
                success=False,
                output="",
                errors=[str(e)],
                confidence=0.0,
                agent_name=self.name,
                duration_sec=time.time() - start_time
            )
    
    async def _generate_content(
        self,
        task: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to generate content based on task and context.
        
        Returns:
            Dict with content, title, metadata
        """
        # Build context string
        context_str = ""
        if context:
            # Check for researcher output
            if "researcher_output" in context:
                context_str = f"\n\nRESEARCH FINDINGS:\n{context['researcher_output']}"
            else:
                context_str = f"\n\nCONTEXT:\n{json.dumps(context, indent=2)}"
        
        user_prompt = f"""WRITING TASK:
{task}
{context_str}

Create professional content based on the task and context.
Return JSON only."""
        
        try:
            response = await call_llm(
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.7  # Higher temperature for creative writing
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
            
            if start != -1 and end != -1:
                response_text = response_text[start:end+1]
            
            content_data = json.loads(response_text)
            
            logger.debug(
                "writer_content_generated",
                title=content_data.get("title"),
                format=content_data.get("format")
            )
            
            return content_data
        
        except Exception as e:
            logger.error("writer_generation_error", error=str(e))
            return None