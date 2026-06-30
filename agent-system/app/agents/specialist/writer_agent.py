import json
import structlog
import time
from typing import Dict, Any, Optional

from app.utils.json_parser import extract_json
from app.agents.base_agent import BaseAgent, AgentResult
from app.utils.llm import call_llm_with_system
from app.utils.file_manager import FileManager
from app.utils.output_formatter import get_formatter
from app.core.config import settings

logger = structlog.get_logger()


class WriterAgent(BaseAgent):
    """
    Specialist agent for content creation.

    Takes information (often from ResearcherAgent) and produces
    well-formatted content, then saves it to the workspace via
    OutputFormatter — making it easy to support new file formats
    (markdown, HTML, PDF) without changing this agent.

    Current supported formats (set "format" in LLM response):
      txt / text / article / blog / summary / report / email / outline
      markdown / md
    """

    SYSTEM_PROMPT = """You are a professional content writer. Your job is to create clear, engaging content.

Given a task and context (often research findings), you:
1. Understand the requirements (article, summary, blog post, report, etc.)
2. Structure the content appropriately
3. Write in a clear, professional style
4. Format for readability using markdown where appropriate

INPUT:
- Task description (what to write)
- Context (research findings, data, etc.)

OUTPUT FORMAT (JSON only):
{
  "content": "the full written content (use markdown formatting)",
  "title": "content title",
  "word_count": <estimated integer word count>,
  "format": "article|blog|summary|report|email|outline|markdown",
  "save_to_file": true,
  "filename": "descriptive_filename"
}

Rules:
- Always set save_to_file to true and provide a meaningful filename (no extension needed)
- word_count should be your honest estimate of the actual words in content
- If research findings are provided, draw from them — don't make things up
- Keep content focused and complete
- Use "markdown" as format when the content benefits from structured headings/lists

RESPOND ONLY WITH JSON."""

    def __init__(
        self, name: str = "writer_001", model: str = "claude-haiku-4-5-20251001"
    ):
        super().__init__(name=name, role="writer", allowed_tools=["file_write"])
        self.model = model
        self._fm   = FileManager(base_dir=settings.WORKSPACE_DIR)

    async def execute(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        start_time = time.time()
        logger.info("writer_executing", task=task)

        try:
            content_result = await self._generate_content(task, context)

            if not content_result:
                self.record_failure()
                return AgentResult(
                    success=False,
                    output="",
                    errors=["Failed to generate content"],
                    confidence=0.0,
                    agent_name=self.name,
                    duration_sec=time.time() - start_time,
                )

            content        = content_result.get("content", "")
            title          = content_result.get("title", "Untitled")
            word_count     = content_result.get("word_count") or len(content.split())
            content_format = content_result.get("format", "txt")
            filename       = content_result.get("filename", "output")
            save_to_file   = content_result.get("save_to_file", True)

            # Human-readable preview (always plain text regardless of format)
            output = f"# {title}\n\n{content}\n\n---\nFormat: {content_format} | Words: ~{word_count}"

            # Delegate file writing to the appropriate formatter
            saved_file = ""
            if save_to_file and filename:
                formatter                = get_formatter(content_format)
                saved_file, char_count   = formatter.write(
                    file_manager=self._fm,
                    filename=filename,
                    title=title,
                    content=content,
                )
                if saved_file:
                    output += f"\n\nwrote {char_count} characters to {saved_file}"
                else:
                    logger.warning("writer_file_save_failed", filename=filename)

            duration = time.time() - start_time
            logger.info(
                "writer_completed",
                title=title,
                word_count=word_count,
                format=content_format,
                saved_file=saved_file,
                duration=duration,
            )

            self.record_success()
            return AgentResult(
                success=True,
                output=output,
                metadata={
                    "title":      title,
                    "word_count": word_count,
                    "format":     content_format,
                    "filename":   saved_file,
                },
                confidence=0.8,
                agent_name=self.name,
                duration_sec=duration,
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
                duration_sec=time.time() - start_time,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _generate_content(
        self, task: str, context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        context_str = ""
        if context:
            if "researcher_output" in context:
                context_str = f"\n\nRESEARCH FINDINGS:\n{context['researcher_output'][:2000]}"
            elif "week4_output" in context:
                context_str = f"\n\nRESEARCH FINDINGS:\n{context['week4_output'][:2000]}"
            elif "search_results" in context:
                context_str = f"\n\nSEARCH RESULTS:\n{context['search_results'][:2000]}"
            else:
                safe = {k: v for k, v in context.items()
                        if k in ("task_description", "session_history")}
                if safe:
                    context_str = (
                        f"\n\nCONTEXT:\n"
                        f"{json.dumps(safe, indent=2, default=str)[:1000]}"
                    )

        user_prompt = (
            f"WRITING TASK:\n{task}"
            f"{context_str}\n\n"
            "Create professional content. Return JSON only."
        )

        try:
            response = await call_llm_with_system(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=self.model,
                temperature=0.7,
            )

            content_data = extract_json(response, context="writer")
            if not content_data:
                logger.error(
                    "writer_json_error",
                    response_preview=(response or "")[:200],
                )
                return None

            return content_data

        except Exception as e:
            logger.error("writer_generation_error", error=str(e))
            return None
