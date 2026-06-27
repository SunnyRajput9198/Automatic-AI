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
        # Find outermost JSON object by brace matching
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        depth = 0
        start_idx = end_idx = -1

        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start_idx != -1:
                    end_idx = i
                    break

        if start_idx != -1 and end_idx != -1:
            try:
                return json.loads(text[start_idx : end_idx + 1])
            except json.JSONDecodeError as e:
                logger.error(
                    "json_parser_failed",
                    context=context,
                    error=str(e),
                    preview=text[start_idx : start_idx + 200],
                )

    logger.error("json_parser_no_json_found", context=context, preview=text[:150])
    return None