"""Bounded evidence review and research-plan state for multi-source tasks."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.utils.json_parser import extract_json
from app.utils.llm import call_openai_with_system, capture_llm_attempts

RESEARCH_TOOLS = frozenset({
    "web_search", "web_fetch", "github", "news_search",
    "semantic_scholar_search", "wikipedia_search",
})


class ResearchStep(BaseModel):
    step: int
    subquestion: str
    tool: str
    instruction: str
    depends_on: list[int] = Field(default_factory=list)
    status: str = "pending"
    evidence_ids: list[int] = Field(default_factory=list)


class ResearchConflict(BaseModel):
    claim: str
    evidence_refs: list[int] = Field(default_factory=list)
    explanation: str
    resolved: bool = False


class FollowUpQuery(BaseModel):
    subquestion: str
    tool: str
    instruction: str


class ResearchAssessment(BaseModel):
    sufficient: bool
    addressed_subquestions: list[int] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    conflicts: list[ResearchConflict] = Field(default_factory=list)
    follow_up_queries: list[FollowUpQuery] = Field(default_factory=list)
    stopping_reason: str = ""


class ResearchPlan(BaseModel):
    objective: str
    subquestions: list[str]
    required_capabilities: list[str]
    capabilities_by_subquestion: dict[str, list[str]] = Field(default_factory=dict)
    research_steps: list[ResearchStep]
    completed_steps: list[int] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    conflicts: list[ResearchConflict] = Field(default_factory=list)
    iteration_count: int = 1
    stopping_reason: str = "initial_plan_executed"

    @classmethod
    def from_intent(cls, objective: str, steps: list[dict[str, Any]], intent: dict[str, Any]) -> "ResearchPlan":
        proposed = intent.get("research")
        questions = []
        capabilities_by_subquestion: dict[str, list[str]] = {}
        if isinstance(proposed, dict):
            raw_questions = proposed.get("subquestions", [])
            if isinstance(raw_questions, list):
                questions = [str(q.get("question", "")).strip() if isinstance(q, dict) else str(q).strip()
                             for q in raw_questions]
                for raw in raw_questions:
                    if isinstance(raw, dict) and isinstance(raw.get("question"), str):
                        capabilities_by_subquestion[raw["question"].strip()] = [
                            tool for tool in raw.get("required_capabilities", [])
                            if isinstance(tool, str) and tool in RESEARCH_TOOLS
                        ]
        questions = [q for q in questions if q]
        research_steps = []
        step_questions = questions[:]
        for step in steps:
            instruction = step["instruction"]
            # Plans that do not include explicit decomposition still retain
            # the planner's dynamically selected research question per step.
            subquestion = step.get("subquestion") or instruction
            if step_questions:
                # Map instruction text back to a declared question where
                # possible. Capability-only matching is ambiguous when the
                # same wrapper serves several questions.
                subquestion = _match_subquestion(
                    step.get("subquestion") or instruction, step["tool"],
                    questions, capabilities_by_subquestion,
                ) or subquestion
            research_steps.append(ResearchStep(
                step=step["step"], subquestion=subquestion, tool=step["tool"],
                instruction=instruction, depends_on=step.get("depends_on", []),
            ))
            capabilities_by_subquestion.setdefault(subquestion, [])
            if step["tool"] in RESEARCH_TOOLS and step["tool"] not in capabilities_by_subquestion[subquestion]:
                capabilities_by_subquestion[subquestion].append(step["tool"])
            if not questions:
                questions.append(subquestion)
        return cls(
            objective=objective,
            subquestions=questions,
            required_capabilities=list(dict.fromkeys(
                [s.tool for s in research_steps]
                + ([tool for q in proposed.get("subquestions", []) if isinstance(q, dict)
                    for tool in q.get("required_capabilities", [])] if isinstance(proposed, dict) else [])
            )),
            capabilities_by_subquestion=capabilities_by_subquestion,
            research_steps=research_steps,
        )

    def record_step(self, step_number: int, status: str, evidence_id: int | None = None) -> None:
        item = next((s for s in self.research_steps if s.step == step_number), None)
        if item is None:
            return
        item.status = status
        if evidence_id is not None and evidence_id not in item.evidence_ids:
            item.evidence_ids.append(evidence_id)
        if status == "completed" and step_number not in self.completed_steps:
            self.completed_steps.append(step_number)

    def append_followups(self, assessment: ResearchAssessment, seen: set[str], max_steps: int) -> list[dict[str, Any]]:
        """Validate new steps and reject repeated queries/tools before execution."""
        added = []
        existing_count = len(self.research_steps)
        for followup in assessment.follow_up_queries:
            if len(added) >= max_steps:
                break
            if followup.tool not in RESEARCH_TOOLS or not followup.instruction.strip():
                continue
            key = normalize_research_key(followup.tool, followup.instruction)
            if key in seen:
                continue
            seen.add(key)
            number = existing_count + len(added) + 1
            item = ResearchStep(
                step=number, subquestion=followup.subquestion.strip(),
                tool=followup.tool, instruction=followup.instruction.strip(),
            )
            self.research_steps.append(item)
            if item.subquestion not in self.subquestions:
                self.subquestions.append(item.subquestion)
            mapped = self.capabilities_by_subquestion.setdefault(item.subquestion, [])
            if item.tool not in mapped:
                mapped.append(item.tool)
            self.required_capabilities.append(item.tool)
            added.append({
                "step": number, "tool": item.tool, "instruction": item.instruction,
                "reasoning": "Research follow-up for an unresolved information gap",
                "depends_on": [], "fallback_tools": [],
            })
        self.required_capabilities = list(dict.fromkeys(self.required_capabilities))
        return added


def normalize_research_key(tool: str, instruction: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", instruction.lower()).strip()
    return f"{tool}:{normalized}"


def _match_subquestion(
    instruction: str, tool: str, questions: list[str], capabilities: dict[str, list[str]],
) -> str | None:
    words = lambda value: set(re.findall(r"[a-z0-9]+", value.lower())) - {
        "what", "which", "how", "when", "where", "the", "and", "for", "from", "with", "about", "does", "are", "is", "to", "of",
    }
    instruction_words = words(instruction)
    ranked = []
    for question in questions:
        question_words = words(question)
        if not question_words:
            continue
        common = len(instruction_words & question_words)
        overlap = common / min(len(instruction_words), len(question_words))
        capability_bonus = 0.08 if tool in capabilities.get(question, []) else 0
        ranked.append((overlap + capability_bonus, overlap, common, question))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    score, overlap, common, question = ranked[0]
    if common >= 2 and overlap >= 0.5 and (len(ranked) == 1 or score > ranked[1][0]):
        return question
    # Capability is a reliable mapping only when it identifies one question.
    matching = [q for q in questions if tool in capabilities.get(q, [])]
    return matching[0] if len(matching) == 1 else None


def research_stop_reason(
    assessment: ResearchAssessment, *, iteration: int, max_iterations: int,
    tool_calls: int, max_tool_calls: int, elapsed_seconds: float, max_seconds: float,
) -> str | None:
    """Return the policy stopping reason, or None when another iteration is allowed."""
    if assessment.sufficient:
        return assessment.stopping_reason or "evidence_sufficient"
    if elapsed_seconds >= max_seconds:
        return "research_time_limit"
    if iteration >= max_iterations:
        return "maximum_research_iterations_reached"
    if tool_calls >= max_tool_calls:
        return "maximum_tool_calls_reached"
    if not assessment.follow_up_queries:
        return assessment.stopping_reason or "useful_sources_exhausted"
    return None


def unreviewed_coverage_assessment(
    plan: ResearchPlan, evidence: list[dict[str, Any]], reason: str,
) -> ResearchAssessment:
    """Conservative coverage fallback when the LLM reviewer is unavailable.

    It only calls a planned subquestion unaddressed when no successful planned
    step is mapped to that subquestion. It does not claim that successful
    evidence is relevant or sufficient; it requests targeted evidence for
    clearly unexecuted gaps and otherwise records the review limitation.
    """
    addressed = set()
    for item in evidence:
        if item.get("status") == "completed" and any(
            attempt.get("status") == "success" and (attempt.get("result") or "").strip()
            for attempt in item.get("attempts", [])
        ):
            addressed.add(item.get("subquestion"))
    gaps = [question for question in plan.subquestions if question not in addressed]
    conflicts = _detect_explicit_conflicts(plan, evidence)
    for conflict in conflicts:
        if conflict.claim not in gaps:
            gaps.append(conflict.claim)
    followups = []
    used_for_question: dict[str, set[str]] = {}
    for item in evidence:
        for attempt in item.get("attempts", []):
            if attempt.get("status") == "success":
                used_for_question.setdefault(item.get("subquestion", ""), set()).add(attempt.get("wrapper", ""))
    for question in gaps[:2]:
        capabilities = plan.capabilities_by_subquestion.get(question, [])
        used = used_for_question.get(question, set())
        candidates = [tool for tool in capabilities if tool in RESEARCH_TOOLS and tool not in used]
        if not candidates:
            # A planner did not map this gap or all mapped sources were already
            # used. Public web search is a reasonable broad research source;
            # exact-authority requirements are still enforced by executor rules.
            candidates = ["web_search"]
        tool = candidates[0]
        followups.append(FollowUpQuery(
            subquestion=question,
            tool=tool,
            instruction=f"Targeted research for the unaddressed subquestion: {question}. Use {tool} and return source-attributed evidence.",
        ))
    return ResearchAssessment(
        sufficient=False,
        addressed_subquestions=[i for i, question in enumerate(plan.subquestions, 1) if question in addressed],
        unresolved_questions=gaps,
        conflicts=conflicts,
        follow_up_queries=followups,
        stopping_reason=f"Evidence reviewer unavailable ({reason}); only unexecuted planned gaps were retried",
    )


def _detect_explicit_conflicts(plan: ResearchPlan, evidence: list[dict[str, Any]]) -> list[ResearchConflict]:
    """Flag clear positive/negative source statements without inferring resolution."""
    positive = re.compile(r"\b(?:supports?|supported|available|enabled|works|can resume|can restart)\b", re.I)
    negative = re.compile(r"\b(?:does not support|doesn't support|not supported|unsupported|unavailable|not available|disabled|cannot resume|can't resume|cannot restart|can't restart)\b", re.I)
    conflicts = []
    for question in plan.subquestions:
        statements = []
        for item in evidence:
            if item.get("subquestion") != question or item.get("status") != "completed":
                continue
            for attempt in item.get("attempts", []):
                if attempt.get("status") != "success":
                    continue
                result = attempt.get("result") or ""
                is_negative = bool(negative.search(result))
                is_positive = not is_negative and bool(positive.search(result))
                if is_negative or is_positive:
                    statements.append((item.get("evidence_id"), is_positive))
        polarities = {polarity for _, polarity in statements}
        refs = list(dict.fromkeys(ref for ref, _ in statements if ref is not None))
        if len(polarities) > 1 and len(refs) > 1:
            conflicts.append(ResearchConflict(
                claim=question,
                evidence_refs=refs,
                explanation="Retrieved sources contain explicit positive and negative statements; the conflict could not be resolved while evidence review was unavailable.",
                resolved=False,
            ))
    return conflicts


def make_research_assessment(payload: Any, subquestion_count: int, evidence_count: int) -> ResearchAssessment:
    """Validate model output and bind conflict references to actual evidence ids."""
    if not isinstance(payload, dict):
        raise ValueError("Research assessment must be an object")
    payload = dict(payload)
    raw_unresolved = payload.get("unresolved_questions", [])
    if isinstance(raw_unresolved, list):
        unresolved = []
        for item in raw_unresolved:
            if isinstance(item, str):
                unresolved.append(item)
                continue
            if not isinstance(item, dict):
                raise ValueError("Unresolved research gaps must be strings or question objects")
            question = next((item.get(key) for key in ("question", "subquestion", "gap", "issue")
                             if isinstance(item.get(key), str) and item[key].strip()), None)
            detail = next((item.get(key) for key in ("reason", "why", "explanation", "needed_information")
                           if isinstance(item.get(key), str) and item[key].strip()), None)
            if not question:
                raise ValueError("Unresolved research gap object has no question text")
            unresolved.append(f"{question.strip()}: {detail.strip()}" if detail else question.strip())
        payload["unresolved_questions"] = unresolved
    assessment = ResearchAssessment.model_validate(payload)
    assessment.addressed_subquestions = [
        idx for idx in assessment.addressed_subquestions if 1 <= idx <= subquestion_count
    ]
    uncovered = [idx for idx in range(1, subquestion_count + 1)
                 if idx not in assessment.addressed_subquestions]
    if uncovered:
        assessment.unresolved_questions.extend(
            f"Subquestion {idx} has not been verified as addressed" for idx in uncovered
        )
    assessment.conflicts = [
        conflict for conflict in assessment.conflicts
        if conflict.claim.strip()
    ]
    for conflict in assessment.conflicts:
        conflict.evidence_refs = [idx for idx in conflict.evidence_refs if 1 <= idx <= evidence_count]
    assessment.follow_up_queries = [
        query for query in assessment.follow_up_queries
        if query.tool in RESEARCH_TOOLS and query.subquestion.strip() and query.instruction.strip()
    ][:6]
    # Unresolved gaps or unresolved conflicts cannot be called sufficient,
    # regardless of an internally inconsistent model boolean.
    if assessment.unresolved_questions or any(not conflict.resolved for conflict in assessment.conflicts):
        assessment.sufficient = False
    return assessment


class ResearchSufficiencyReviewer:
    """Review coverage, relevance, failures and conflicts from collected evidence."""

    SYSTEM_PROMPT = """You review a bounded research run. Decide whether the user's objective can now be answered reliably.
Assess coverage against every subquestion, whether successful evidence is relevant and supports conclusions, source diversity only where it improves reliability, failed important sources, and contradictory claims. Do not equate result count with sufficiency. Do not invent sources, findings, conflicts, or tool failures.
Return only JSON with this schema:
{"sufficient":true,"addressed_subquestions":[1],"unresolved_questions":[],"conflicts":[{"claim":"...","evidence_refs":[1,2],"explanation":"...","resolved":false}],"follow_up_queries":[{"subquestion":"...","tool":"web_search","instruction":"..."}],"stopping_reason":"..."}
Keep explanations and stopping_reason concise so the structured decision fits in the response budget.
Use evidence_refs as one-based evidence item IDs. If more evidence is materially useful, set sufficient=false and provide one or more distinct targeted follow-up queries using only the available research wrappers. Avoid repeating an existing query. If useful sources are exhausted or more research is unlikely to help, leave follow_up_queries empty and explain the limitation in stopping_reason. A conflict is resolved only when source quality, recency, or direct evidence clearly reconciles it; otherwise preserve it as unresolved."""

    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model
        self.last_call_diagnostics: list[dict[str, Any]] = []

    async def assess(self, plan: ResearchPlan, evidence: list[dict[str, Any]]) -> ResearchAssessment:
        prompt = (
            f"Objective:\n{plan.objective}\n\n"
            f"Subquestions (one-based):\n{json_list(plan.subquestions)}\n\n"
            f"Research plan and evidence:\n{json_list(_compact_evidence_for_review(evidence))}\n\n"
            "Return a sufficiency assessment grounded only in this material."
        )
        last_error: Exception | None = None
        self.last_call_diagnostics = []
        for attempt in range(2):
            with capture_llm_attempts() as call_diagnostics:
                try:
                    response = await call_openai_with_system(
                        system_prompt=self.SYSTEM_PROMPT,
                        user_prompt=prompt if attempt == 0 else prompt + "\nYour last response was malformed. Return the exact JSON schema.",
                        model=self.model, temperature=0.1, max_tokens=3000,
                        reasoning_effort="low",
                    )
                except (ValueError, TypeError, ValidationError) as exc:
                    self.last_call_diagnostics.extend(call_diagnostics)
                    last_error = exc
                    continue
                except Exception as exc:
                    self.last_call_diagnostics.extend(call_diagnostics)
                    # The LLM helper already performs its own transport retries.
                    # Preserve successful tool evidence if the reviewer service is
                    # temporarily unavailable instead of failing the whole task.
                    return unreviewed_coverage_assessment(plan, evidence, type(exc).__name__)
            self.last_call_diagnostics.extend(call_diagnostics)
            try:
                payload = extract_json(response, context="research_sufficiency")
                assessment = make_research_assessment(payload, len(plan.subquestions), len(evidence))
                if self.last_call_diagnostics:
                    self.last_call_diagnostics[-1]["parser_status"] = "valid"
                return assessment
            except ValidationError as exc:
                if self.last_call_diagnostics:
                    self.last_call_diagnostics[-1].update({
                        "parser_status": "schema_invalid",
                        "parser_error_types": list(dict.fromkeys(item.get("type", "unknown") for item in exc.errors())),
                        "parser_error_locations": [
                            ".".join(str(part) for part in item.get("loc", ()))
                            for item in exc.errors()
                        ][:12],
                    })
                last_error = exc
            except (ValueError, TypeError) as exc:
                if self.last_call_diagnostics:
                    self.last_call_diagnostics[-1].update({
                        "parser_status": "invalid_json",
                        "parser_error_type": type(exc).__name__,
                    })
                last_error = exc
        # Fail closed on sufficiency: do not claim evidence is enough when the
        # evaluator failed. The caller will stop at configured limits.
        return unreviewed_coverage_assessment(
            plan, evidence, f"invalid output after retry ({type(last_error).__name__ if last_error else 'unknown'})"
        )


def json_list(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, default=str)


def _compact_evidence_for_review(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep provenance and useful excerpts while bounding evaluator input size."""
    compact = []
    excerpt_budget = 9000
    attempt_budget = 8
    for item in evidence:
        attempts = []
        for attempt in item.get("attempts", []):
            if attempt_budget <= 0:
                break
            attempt_budget -= 1
            result = str(attempt.get("result") or "")
            excerpt = result[:min(1500, excerpt_budget)]
            excerpt_budget -= len(excerpt)
            attempts.append({
                key: str(attempt.get(key))[:300]
                for key in ("wrapper", "status", "query", "url", "source", "error")
                if attempt.get(key) is not None
            } | {"result_excerpt": excerpt})
        compact.append({
            key: item.get(key)
            for key in ("evidence_id", "subquestion", "source_type", "query", "url", "timestamp", "status", "failure")
            if item.get(key) is not None
        } | {"attempts": attempts})
        if attempt_budget <= 0 or excerpt_budget <= 0:
            break
    return compact


def compact_research_evidence(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bound synthesis evidence while preserving source, status and useful excerpts."""
    compact = []
    excerpt_budget = 12000
    attempt_budget = 12
    for item in evidence:
        record = {
            key: str(item.get(key))[:500]
            for key in ("evidence_id", "subquestion", "instruction", "tool", "status", "error")
            if item.get(key) not in (None, "")
        }
        attempts = []
        for attempt in item.get("attempts", []):
            if attempt_budget <= 0 or excerpt_budget <= 0:
                break
            attempt_budget -= 1
            result = str(attempt.get("result") or attempt.get("extracted_information") or "")
            excerpt = result[:min(1500, excerpt_budget)]
            excerpt_budget -= len(excerpt)
            attempts.append({
                key: str(attempt.get(key))[:500]
                for key in ("source", "source_type", "wrapper", "status", "query", "source_url", "url", "retrieved_at", "failure_type", "error")
                if attempt.get(key) not in (None, "")
            } | {"result_excerpt": excerpt})
        record["attempts"] = attempts
        if not attempts and excerpt_budget > 0:
            result = str(item.get("result") or item.get("extracted_information") or "")
            if result:
                excerpt = result[:min(1500, excerpt_budget)]
                excerpt_budget -= len(excerpt)
                record["result_excerpt"] = excerpt
        compact.append(record)
        if attempt_budget <= 0 or excerpt_budget <= 0:
            break
    return compact
