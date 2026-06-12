import structlog
from typing import Dict, Any, List

logger = structlog.get_logger()


class ReferenceResolver:
    """
    Resolves conversational references such as:
    - it
    - that
    - those
    - them
    - which one
    - best one
    - compare it
    - compare them
    """

    REFERENCE_PATTERNS = [
        "which one",
        "which is best",
        "best one",
        "compare it",
        "compare them",
        "from the research",
        "from that research",
        "it",
        "that",
        "those",
        "them",
    ]

    def has_reference(self, query: str) -> bool:
        """
        Detect whether query refers to previous context.
        """
        query_lower = query.lower()

        return any(pattern in query_lower for pattern in self.REFERENCE_PATTERNS)

    def resolve(
        self, query: str, session_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Resolve reference using most recent task.
        Version 1: use last task only.
        """

        if not session_history:
            return {
                "has_reference": False,
                "resolved_subject": None,
                "enriched_query": query,
            }

        previous_task = None

        for item in reversed(session_history):
            if item["task"].lower().strip() != query.lower().strip():
                previous_task = item
                break

        if previous_task is None:
            return {
                "has_reference": False,
                "resolved_subject": None,
                "enriched_query": query,
            }

        task_text = previous_task.get("task", "")
        output_text = previous_task.get("output", "")[:150]

        entities = previous_task.get("entities", [])

        entity_names = [
            e["name"] for e in entities if isinstance(e, dict) and e.get("name")
        ]

        logger.info("reference_entities_found", entities=entity_names)

        enriched_query = f"""
CURRENT QUESTION:
{query}

PREVIOUS TASK:
{task_text}

EXTRACTED ENTITIES:
{", ".join(entity_names)}

PREVIOUS FINDINGS:
{output_text}

INSTRUCTION:
If the user says:
- which one
- best one
- compare them
- compare it
- which is best

assume they are referring to the EXTRACTED ENTITIES above.

Use those entities as the comparison candidates.

Do not perform a new search if the answer can be derived from the previous task.
"""

        logger.info(
            "reference_resolved",
            original_query=query,
            previous_task=task_text,
        )

        return {
            "has_reference": True,
            "resolved_subject": task_text,
            "enriched_query": enriched_query,
        }
