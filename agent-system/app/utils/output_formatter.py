"""
OutputFormatter — modular file output for WriterAgent.

Adding a new format (e.g. PDF, HTML) means:
  1. Subclass OutputFormatter
  2. Implement render() and file_extension
  3. Register in FORMATTERS
  4. The LLM's "format" field in the writing response selects it automatically.

No changes to WriterAgent are needed.
"""

from __future__ import annotations

import re
import structlog
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from app.utils.file_manager import FileManager

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class OutputFormatter(ABC):
    """
    Abstract base for all output formatters.

    Responsibilities:
    - Render content + title into the final file body (render)
    - Declare the file extension (file_extension)
    - Write the rendered output to the workspace (write)

    WriterAgent calls write() and receives (saved_filename, char_count).
    """

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension WITHOUT the leading dot, e.g. 'txt' or 'md'."""

    @abstractmethod
    def render(self, title: str, content: str) -> str:
        """
        Combine title and content into the final file body.
        Return the complete string that will be written to disk.
        """

    def write(
        self,
        file_manager: FileManager,
        filename: str,
        title: str,
        content: str,
    ) -> Tuple[str, int]:
        """
        Render and write content to the workspace.

        Returns:
            (final_filename, char_count) on success
            ("", 0) on failure
        """
        # Normalise extension
        final_filename = self._normalise_filename(filename)
        rendered       = self.render(title, content)

        ok = file_manager.write_file(final_filename, rendered)
        if ok:
            logger.info(
                "output_formatter_saved",
                filename=final_filename,
                formatter=self.__class__.__name__,
                chars=len(rendered),
            )
            return final_filename, len(rendered)

        logger.warning(
            "output_formatter_save_failed",
            filename=final_filename,
            formatter=self.__class__.__name__,
        )
        return "", 0

    def _normalise_filename(self, filename: str) -> str:
        """Strip any existing extension and add the correct one."""
        base = re.sub(r"\.[a-zA-Z0-9]+$", "", filename).strip("_").strip()
        return f"{base}.{self.file_extension}"


# ---------------------------------------------------------------------------
# Concrete formatters
# ---------------------------------------------------------------------------

class TxtFormatter(OutputFormatter):
    """
    Plain-text output.  Current default — preserves existing behaviour exactly.
    """

    @property
    def file_extension(self) -> str:
        return "txt"

    def render(self, title: str, content: str) -> str:
        return f"# {title}\n\n{content}"


class MarkdownFormatter(OutputFormatter):
    """
    Markdown output with a YAML-style front matter block.
    Ready to use — just pass format='markdown' from the LLM response.
    """

    @property
    def file_extension(self) -> str:
        return "md"

    def render(self, title: str, content: str) -> str:
        # Normalise title for front-matter (strip markdown heading chars)
        clean_title = title.lstrip("#").strip()
        front_matter = f"---\ntitle: \"{clean_title}\"\n---\n\n"
        # Ensure the content starts with a top-level heading
        if not content.lstrip().startswith("#"):
            return f"{front_matter}# {clean_title}\n\n{content}"
        return f"{front_matter}{content}"


# ---------------------------------------------------------------------------
# Registry — maps LLM "format" field values to formatter instances
# ---------------------------------------------------------------------------

FORMATTERS: Dict[str, OutputFormatter] = {
    "txt":      TxtFormatter(),
    "text":     TxtFormatter(),
    "article":  TxtFormatter(),
    "blog":     TxtFormatter(),
    "summary":  TxtFormatter(),
    "report":   TxtFormatter(),
    "email":    TxtFormatter(),
    "outline":  TxtFormatter(),
    "markdown": MarkdownFormatter(),
    "md":       MarkdownFormatter(),
}

_DEFAULT_FORMATTER = TxtFormatter()


def get_formatter(format_name: Optional[str]) -> OutputFormatter:
    """
    Return the formatter for the given format name.
    Falls back to TxtFormatter for unknown or None values.
    """
    if not format_name:
        return _DEFAULT_FORMATTER
    return FORMATTERS.get(format_name.lower().strip(), _DEFAULT_FORMATTER)
