import json
import structlog
from typing import Any, Optional

logger = structlog.get_logger()

def extract_json(response: str, context: str = "") -> Optional[dict]:
    """
    Extract and parse JSON from LLM response.
    Handles:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace
    - JSON embedded in surrounding text

    Args:
        response: Raw LLM response string
        context:  Optional label for error logging (e.g. "planner", "critic")

    Returns:
        Parsed dict, or None if parsing fails
    """
    if not response:
        return None

    text = response.strip()#remove all trailing or leading whitespaces

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening ```json or ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Decode from each object boundary. JSONDecoder tracks quoted strings and
    # escapes correctly, unlike a brace counter (which breaks on `{` or `}` in
    # explanations, code, and source excerpts).
    decoder = json.JSONDecoder()
    for start_idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text, start_idx)
        except json.JSONDecodeError as exc:
            logger.debug(
                "json_parser_candidate_rejected",
                context=context,
                position=exc.pos,
                error=exc.msg,
            )
            continue
        if isinstance(parsed, dict):
            return parsed

    logger.error("json_parser_no_json_found", context=context, preview=text[:150])
    return None
