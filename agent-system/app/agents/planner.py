"""Semantic planner for direct answers and minimum-capability tool plans."""
from typing import Any, Dict, List
import re

import structlog

from app.utils.json_parser import extract_json
from app.utils.llm import call_openai_with_system

logger = structlog.get_logger()
_VALID_TOOLS = {
    "web_search", "web_fetch", "github", "get_weather", "news_search",
    "semantic_scholar_search", "wikipedia_search", "python_executor",
    "shell_executor", "file_read", "file_write", "file_append", "file_list", "file_delete",
}


class PlannerAgent:
    """Classify the requested outcome and produce only the needed tool steps."""

    SYSTEM_PROMPT = """You plan how to answer a user's request. Return one JSON object only:
{"mode":"direct_response|tool_execution|multi_tool","answer":"plain human-readable text, never nested JSON","research":null,"steps":[{"step":1,"tool":"web_search","subquestion":"...","instruction":"...","depends_on":[],"reasoning":"...","fallback_tools":[]}]}

Choose direct_response when the request can be answered from the conversation or stable general knowledge without external data or computation. Put the complete answer in `answer` as plain human-readable prose, not JSON, code, or internal intent metadata; use an empty steps list. For requested plans/lists, answer with the requested deliverable directly.

Choose tool_execution when one capability is needed; choose multi_tool only when distinct capabilities or independent sources are genuinely required. `answer` is empty for tool modes. Make one step per necessary capability, no more than 10. Every step MUST name its exact wrapper in the `tool` field and provide a specific executable instruction. `fallback_tools` lists only valid alternative wrappers that can actually provide this same required information; leave it empty when no reliable alternative exists. For independent calls use no dependencies; use dependencies only when a later call needs an earlier result. Do not create/write files unless the user asks. Never call tools just to demonstrate them.

For genuinely complex, multi-dimensional research, set `research` to an object with `objective` and a dynamically derived `subquestions` list. Each subquestion is an object with `question` and `required_capabilities` (wrapper names). The subquestions should decompose the user's actual objective and align with the research steps. For each research step include a `subquestion` field naming the one subquestion it answers; do not assign every step to the first subquestion. For simple lookups or direct answers, set `research` to null. Do not make up a generic checklist.

Available wrappers (use the named wrapper explicitly in each instruction):
- web_search: current/general internet search results; use for fresh or source-backed web information. Inputs query and optional num_results. Do not use for GitHub repository metadata or weather.
- web_fetch: retrieve a specific public HTTPS webpage when URL is known; input url. Do not fetch arbitrary/internal URLs.
- github: query a public GitHub repository's latest commit or latest release. Inputs owner, repo, operation (latest_commit/latest_release), optional branch. Do not use generic search for this information. Private repositories may require configured GitHub token.
- get_weather: current conditions for a city, input city; use for weather and never invent conditions.
- news_search: current news articles for a topic, input query; use for news specifically.
- semantic_scholar_search: primary scholarly index for academic papers and studies, input query. If the user requests papers, scholarly research, or peer-reviewed studies, select this wrapper as the primary academic source. Use web_search only as a clearly labeled fallback if this wrapper fails; web search results are not an equivalent scholarly index.
- wikipedia_search: encyclopedic lookup, input query.
- python_executor: run actual Python code for meaningful calculations/data processing; input code. Skip for trivial arithmetic and ordinary reasoning.
- file_read/file_write/file_append/file_list/file_delete: operate only in the agent workspace when user explicitly requests file operations.
- shell_executor: restricted workspace command when user explicitly needs a shell command and file tools are insufficient.

Use the minimum necessary wrappers. A request for fresh developments needs web_search; a GitHub commit/release request needs github; weather needs get_weather. For a request for both current news and academic papers, select news_search and semantic_scholar_search; add web_search only when broader web announcements are specifically needed or as a fallback. A public GitHub repository may use web_search as a lower-confidence fallback only when public visibility is explicit; never use that fallback for private or visibility-unknown repositories. For academic paper lookup, web_search can supplement a failed Semantic Scholar call. Use fallback_tools only for these source-compatible alternatives and leave it empty otherwise. If the repository owner/name is not available in the user request or conversation, ask for it in a direct_response instead of searching the web or inventing a repository. A request requiring both sources needs both calls, then the application will synthesize their results. If only a direct answer is needed, do not add steps. Output valid JSON with the exact mode enum."""

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model

    @staticmethod
    def _mandatory_tools(user_task: str) -> list[str]:
        """Safety gate for unmistakable live-data and scholarly capabilities."""
        text = user_task.lower()
        selected = []
        if re.search(r"\b(weather|forecast|temperature|humidity|rainfall)\b", text):
            selected.append("get_weather")
        if re.search(r"\b(github|github\.com)\b", text) and re.search(r"\b(commit|release|repository|repo)\b", text):
            selected.append("github")
        if re.search(r"\b(academic|scholarly|peer[ -]reviewed|research papers?|papers? (?:about|on|for|related)|scientific studies)\b", text):
            selected.append("semantic_scholar_search")
        if re.search(r"\b(latest|recent|current|breaking)\s+(?:news|headlines)\b", text):
            selected.append("news_search")
        elif not ("semantic_scholar_search" in selected) and re.search(
            r"\b(latest developments|current developments|search for|look up|find recent information)\b", text
        ):
            selected.append("web_search")
        return list(dict.fromkeys(selected))

    @staticmethod
    def _ensure_required_capabilities(steps: list[dict], required: list[str], user_task: str) -> list[dict]:
        steps = [dict(step) for step in steps]
        present = {step["tool"] for step in steps}
        for tool in required:
            if tool in present:
                continue
            instruction = (
                f"Search Semantic Scholar for recent papers relevant to: {user_task}"
                if tool == "semantic_scholar_search" else
                f"Retrieve the requested current weather using the location in: {user_task}"
                if tool == "get_weather" else
                f"Retrieve the GitHub commit or release requested in: {user_task}"
                if tool == "github" else
                f"Search recent news for: {user_task}"
                if tool == "news_search" else
                f"Search the web for current information about: {user_task}"
            )
            # Replace an incorrectly chosen broad-search step when it is the
            # only planned step. Otherwise preserve distinct capabilities.
            replaceable = next((s for s in steps if s["tool"] not in required), None)
            if replaceable and len(steps) == 1:
                replaceable["tool"] = tool
                replaceable["instruction"] = instruction
            else:
                steps.append({
                    "step": len(steps) + 1, "tool": tool, "instruction": instruction,
                    "reasoning": "Required capability for this explicit information request",
                    "depends_on": [], "fallback_tools": [],
                })
            present.add(tool)
        for index, step in enumerate(steps, 1):
            step["step"] = index
            step["depends_on"] = [dep for dep in step.get("depends_on", []) if dep < index]
        return steps

    @staticmethod
    def _validate_steps(raw_steps: Any, log_key: str) -> List[Dict[str, Any]]:
        if not isinstance(raw_steps, list):
            raise ValueError(f"{log_key} steps must be a list")
        if len(raw_steps) > 10:
            raise ValueError(f"{log_key} plan exceeds the 10-step limit")
        validated: List[Dict[str, Any]] = []
        for index, raw in enumerate(raw_steps, 1):
            if not isinstance(raw, dict) or not isinstance(raw.get("instruction"), str):
                raise ValueError(f"{log_key} step {index} requires an instruction")
            instruction = raw["instruction"].strip()
            if not instruction:
                raise ValueError(f"{log_key} step {index} has an empty instruction")
            deps = raw.get("depends_on", [])
            if (not isinstance(deps, list)
                    or any(not isinstance(d, int) or isinstance(d, bool) or d < 1 or d >= index for d in deps)):
                raise ValueError(f"{log_key} step {index} has invalid dependencies")
            fallbacks = raw.get("fallback_tools", [])
            if (not isinstance(fallbacks, list) or len(fallbacks) > 2
                    or any(not isinstance(name, str) or name not in _VALID_TOOLS for name in fallbacks)):
                raise ValueError(f"{log_key} step {index} has invalid fallback tools")
            validated.append({
                "step": index,
                "tool": str(raw.get("tool", "")).strip(),
                "subquestion": str(raw.get("subquestion", "")).strip(),
                "instruction": instruction,
                "reasoning": str(raw.get("reasoning", "")),
                "depends_on": deps,
                "fallback_tools": fallbacks,
            })
        return validated

    async def plan_intent(self, user_task: str, planning_context: str = "") -> Dict[str, Any]:
        """Return a validated intent object used by the main orchestrator."""
        user_prompt = f"User request:\n{user_task}\n"
        if planning_context.strip():
            user_prompt += f"\nResearch planning context:\n{planning_context.strip()}\n"
        user_prompt += "\nReturn the intent JSON."
        last_error = None
        for attempt in range(2):
            response = await call_openai_with_system(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt if attempt == 0 else (
                    user_prompt + "\n\nThe prior response did not satisfy the schema: "
                    + str(last_error) + ". Correct the structure; each tool step needs `tool` and `instruction`."
                ),
                model=self.model,
                temperature=0.1,
            )
            try:
                data = extract_json(response, context="planner")
                if not isinstance(data, dict):
                    raise ValueError("Planner returned invalid JSON")
                mode = data.get("mode")
                if mode not in {"direct_response", "tool_execution", "multi_tool"}:
                    raise ValueError("Planner returned an unsupported intent mode")
                steps = self._validate_steps(data.get("steps", []), "planner")
                required_tools = self._mandatory_tools(user_task)
                research = data.get("research")
                if research is not None:
                    if (not isinstance(research, dict) or not isinstance(research.get("objective"), str)
                            or not research["objective"].strip()):
                        raise ValueError("Research plan requires an objective")
                    subquestions = research.get("subquestions")
                    if (not isinstance(subquestions, list) or not subquestions
                            or len(subquestions) > 12
                            or any(not isinstance(q, dict) or not isinstance(q.get("question"), str)
                                   or not q["question"].strip()
                                   or not isinstance(q.get("required_capabilities", []), list)
                                   or any(not isinstance(tool, str) or tool not in _VALID_TOOLS
                                          for tool in q.get("required_capabilities", []))
                                   for q in subquestions)):
                        raise ValueError("Research subquestions are malformed")
                answer = data.get("answer", "")
                if mode == "direct_response" and required_tools:
                    mode = "tool_execution" if len(required_tools) == 1 else "multi_tool"
                    answer = ""
                    steps = self._ensure_required_capabilities([], required_tools, user_task)
                if mode == "direct_response":
                    answer = self._normalise_direct_answer(answer)
                    if not answer.strip():
                        raise ValueError("Direct response plan has no answer")
                    steps = []
                elif not steps:
                    raise ValueError("Tool plan has no valid steps")
                elif any(not step["tool"] for step in steps):
                    raise ValueError("Every tool step must name the selected wrapper")
                elif any(step["tool"] not in _VALID_TOOLS for step in steps):
                    raise ValueError("Planner selected an unknown wrapper")
                elif mode == "tool_execution" and len(steps) > 1:
                    mode = "multi_tool"
                elif mode == "multi_tool" and len(steps) == 1:
                    mode = "tool_execution"
                if mode != "direct_response":
                    required_by_research = []
                    if isinstance(research, dict):
                        required_by_research = list(dict.fromkeys(
                            tool for question in research.get("subquestions", [])
                            for tool in question.get("required_capabilities", [])
                        ))
                        steps = self._ensure_required_capabilities(steps, required_by_research, user_task)
                    # Mandatory live-data/scholarly needs take precedence over
                    # a contradictory generic capability in the model plan.
                    steps = self._ensure_required_capabilities(steps, required_tools, user_task)
                    if "semantic_scholar_search" in (required_tools + required_by_research) and not isinstance(research, dict):
                        research = {
                            "objective": user_task,
                            "subquestions": [{
                                "question": user_task,
                                "required_capabilities": ["semantic_scholar_search"],
                            }],
                        }
                    if len(steps) > 10:
                        raise ValueError("Required capabilities exceed the planner's 10-step limit")
                    if len(steps) > 1:
                        mode = "multi_tool"
                intent = {"mode": mode, "answer": answer.strip() if isinstance(answer, str) else "",
                          "steps": steps, "research": research}
                logger.info("planner_intent_selected", mode=mode, num_steps=len(steps), attempt=attempt + 1)
                return intent
            except (ValueError, TypeError, AttributeError) as exc:
                last_error = exc
                if attempt == 0:
                    logger.warning("planner_schema_retry", error_type=type(exc).__name__)
        raise ValueError(f"Planner response remained invalid after retry: {last_error}")

    @staticmethod
    def _normalise_direct_answer(answer: Any) -> str:
        """Keep model-generated structured intent artifacts out of user replies."""
        import json
        if isinstance(answer, dict):
            parsed = answer
        elif isinstance(answer, str):
            text = answer.strip()
            # Some model completions append a prose-prefixed internal intent
            # dump rather than a JSON object (for example, "Intent JSON:").
            # Keep the user-facing sentence and discard that artifact.
            import re
            text = re.split(r"\s+Intent\s+JSON\s*:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            try:
                parsed = json.loads(text)
            except (ValueError, TypeError):
                # Some model responses prepend a clean answer but append their
                # internal intent object. Strip only that recognizable suffix.
                decoder = json.JSONDecoder()
                for index, char in enumerate(text):
                    if char != "{":
                        continue
                    try:
                        suffix, end = decoder.raw_decode(text[index:])
                    except ValueError:
                        continue
                    if isinstance(suffix, dict) and any(
                        key in suffix for key in ("intent", "original_request", "confidence")
                    ):
                        prefix = text[:index].strip()
                        if prefix:
                            return prefix
                        parsed = suffix
                        break
                else:
                    return text
        else:
            return str(answer) if answer is not None else ""
        if isinstance(parsed, dict):
            for key in ("response", "utterance", "answer", "text", "message", "output"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            if "median" in parsed:
                return f"The median is {parsed['median']}."
            return "\n".join(
                f"{key.replace('_', ' ').capitalize()}: {value}"
                for key, value in parsed.items()
                if key not in {"intent", "confidence", "name", "description"}
            )
        return str(parsed)

    async def _call_llm(self, system_prompt: str, user_prompt: str, log_context: str) -> List[Dict]:
        response = await call_openai_with_system(
            system_prompt=system_prompt, user_prompt=user_prompt,
            model=self.model, temperature=0.1,
        )
        data = extract_json(response, context=log_context)
        if not isinstance(data, dict):
            raise ValueError(f"Failed to parse {log_context} JSON")
        steps = self._validate_steps(data.get("steps", []), log_context)
        if not steps:
            raise ValueError(f"{log_context} produced no valid steps")
        return steps

    async def plan(self, user_task: str) -> List[Dict]:
        """Backward-compatible step-only plan used by the research graph."""
        intent = await self.plan_intent(user_task)
        if intent["mode"] == "direct_response":
            return [{"step": 1, "instruction": f"Answer directly: {intent['answer']}",
                     "reasoning": "Direct response requested", "depends_on": []}]
        return intent["steps"]

    async def replan(self, original_task: str, failed_step: str, error: str) -> List[Dict]:
        prompt = (
            f"ORIGINAL TASK:\n{original_task}\n\nFAILED STEP:\n{failed_step}\n\n"
            f"ERROR:\n{error}\n\nCreate a new plan avoiding the failure. Return JSON with steps."
        )
        return await self._call_llm(
            self.SYSTEM_PROMPT + '\nFor this replan return JSON with a steps array.',
            prompt, "replanner",
        )
