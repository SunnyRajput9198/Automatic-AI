import json
import unittest
from unittest.mock import AsyncMock, patch

from app.agents.planner import PlannerAgent
from app.orchestrator.research import (
    ResearchAssessment,
    ResearchPlan,
    ResearchSufficiencyReviewer,
    compact_research_evidence,
    make_research_assessment,
    normalize_research_key,
    research_stop_reason,
    unreviewed_coverage_assessment,
)


class ResearchPlanTests(unittest.TestCase):
    def test_simple_plan_has_no_unnecessary_research_decomposition(self):
        intent = {"research": None}
        plan = ResearchPlan.from_intent("Get today's weather", [], intent)
        self.assertEqual(plan.research_steps, [])
        self.assertEqual(plan.subquestions, [])

    def test_complex_plan_retains_dynamic_subquestions_and_capabilities(self):
        intent = {"research": {
            "objective": "Compare agent frameworks",
            "subquestions": [
                {"question": "Which frameworks are active?", "required_capabilities": ["github"]},
                {"question": "What do recent papers report?", "required_capabilities": ["semantic_scholar_search"]},
            ],
        }}
        plan = ResearchPlan.from_intent("Compare agent frameworks", [
            {"step": 1, "tool": "github", "instruction": "Check framework repositories", "depends_on": []},
            {"step": 2, "tool": "semantic_scholar_search", "instruction": "Search recent framework papers", "depends_on": []},
        ], intent)
        self.assertEqual(len(plan.subquestions), 2)
        self.assertEqual(set(plan.required_capabilities), {"github", "semantic_scholar_search"})
        self.assertEqual(plan.research_steps[0].subquestion, "Which frameworks are active?")

    def test_planner_step_subquestion_prevents_all_steps_being_attributed_to_first_question(self):
        intent = {"research": {
            "objective": "Compare two frameworks",
            "subquestions": [
                {"question": "What is the architecture?", "required_capabilities": ["web_search"]},
                {"question": "How are deployment options documented?", "required_capabilities": ["web_search"]},
            ],
        }}
        plan = ResearchPlan.from_intent("Compare frameworks", [
            {"step": 1, "tool": "web_search", "subquestion": "What is the architecture?", "instruction": "Search architecture docs", "depends_on": []},
            {"step": 2, "tool": "web_search", "subquestion": "How are deployment options documented?", "instruction": "Search official deployment documentation", "depends_on": []},
        ], intent)
        self.assertEqual([step.subquestion for step in plan.research_steps], [
            "What is the architecture?", "How are deployment options documented?",
        ])

    def test_short_step_subquestion_canonicalizes_to_declared_research_question(self):
        declared = "Determine whether official documentation describes cloud-hosted and self-hosted deployment options, including explicit deployment instructions."
        plan = ResearchPlan.from_intent("Research hosting", [
            {"step": 1, "tool": "web_fetch", "subquestion": "Does official documentation explain deployment options?",
             "instruction": "Fetch official deployment docs", "depends_on": []},
        ], {"research": {"objective": "Research hosting", "subquestions": [
            {"question": declared, "required_capabilities": ["web_fetch"]},
        ]}})
        self.assertEqual(plan.research_steps[0].subquestion, declared)

    def test_duplicate_followup_is_not_added(self):
        plan = ResearchPlan.from_intent("Compare A and B", [
            {"step": 1, "tool": "web_search", "instruction": "Find current comparison of A and B", "depends_on": []},
        ], {})
        seen = {normalize_research_key("web_search", "Find current comparison of A and B")}
        assessment = ResearchAssessment(
            sufficient=False,
            unresolved_questions=["What are deployment differences?"],
            follow_up_queries=[{
                "subquestion": "What are deployment differences?",
                "tool": "web_search",
                "instruction": "Find current comparison of A and B",
            }],
        )
        self.assertEqual(plan.append_followups(assessment, seen, 2), [])

    def test_followup_plan_adds_only_valid_research_capabilities(self):
        plan = ResearchPlan.from_intent("Compare A and B", [], {})
        assessment = ResearchAssessment(sufficient=False, follow_up_queries=[
            {"subquestion": "Deployment", "tool": "web_search", "instruction": "Compare deployment models for A and B"},
            {"subquestion": "Write a file", "tool": "file_write", "instruction": "Save results"},
        ])
        added = plan.append_followups(assessment, set(), 1)
        self.assertEqual(len(added), 1)
        self.assertEqual(added[0]["tool"], "web_search")

    def test_conflicts_reference_only_available_evidence_and_prevent_sufficiency(self):
        assessment = make_research_assessment({
            "sufficient": True,
            "addressed_subquestions": [1, 99],
            "unresolved_questions": [],
            "conflicts": [{"claim": "Supports feature X", "evidence_refs": [1, 2, 99], "explanation": "Sources disagree", "resolved": False}],
            "follow_up_queries": [],
            "stopping_reason": "More evidence may help",
        }, subquestion_count=1, evidence_count=2)
        self.assertFalse(assessment.sufficient)
        self.assertEqual(assessment.addressed_subquestions, [1])
        self.assertEqual(assessment.conflicts[0].evidence_refs, [1, 2])

    def test_unaddressed_subquestion_prevents_sufficiency(self):
        assessment = make_research_assessment({
            "sufficient": True, "addressed_subquestions": [1], "unresolved_questions": [],
            "conflicts": [], "follow_up_queries": [], "stopping_reason": "Apparently enough",
        }, subquestion_count=2, evidence_count=2)
        self.assertFalse(assessment.sufficient)
        self.assertIn("Subquestion 2 has not been verified as addressed", assessment.unresolved_questions)

    def test_reviewer_gap_objects_are_normalized_without_losing_reasons(self):
        assessment = make_research_assessment({
            "sufficient": True, "addressed_subquestions": [1],
            "unresolved_questions": [
                {"question": "Deployment options", "reason": "No official self-hosting page was fetched"},
            ],
            "conflicts": [], "follow_up_queries": [], "stopping_reason": "More evidence needed",
        }, subquestion_count=1, evidence_count=1)
        self.assertFalse(assessment.sufficient)
        self.assertEqual(assessment.unresolved_questions, [
            "Deployment options: No official self-hosting page was fetched",
        ])

    def test_synthesis_evidence_compaction_bounds_payload_and_keeps_source_urls(self):
        evidence = [{"evidence_id": i, "subquestion": "Research topic", "tool": "web_fetch",
                     "status": "completed", "attempts": [{"source": f"Source {i}", "wrapper": "web_fetch",
                     "status": "success", "source_url": f"https://example.com/{i}", "result": "x" * 8000}]}
                    for i in range(1, 20)]
        compact = compact_research_evidence(evidence)
        excerpts = [attempt["result_excerpt"] for item in compact for attempt in item["attempts"]]
        urls = [attempt["source_url"] for item in compact for attempt in item["attempts"]]
        self.assertLessEqual(sum(map(len, excerpts)), 12000)
        self.assertLessEqual(len(excerpts), 12)
        self.assertIn("https://example.com/1", urls)

    def test_stopping_is_coverage_and_budget_based(self):
        sufficient = ResearchAssessment(sufficient=True, stopping_reason="Coverage is complete")
        insufficient = ResearchAssessment(
            sufficient=False, unresolved_questions=["Deployment model"],
            follow_up_queries=[{"subquestion": "Deployment", "tool": "web_search", "instruction": "Check deployment"}],
        )
        self.assertEqual(research_stop_reason(sufficient, iteration=0, max_iterations=2,
            tool_calls=1, max_tool_calls=8, elapsed_seconds=1, max_seconds=30), "Coverage is complete")
        self.assertIsNone(research_stop_reason(insufficient, iteration=0, max_iterations=2,
            tool_calls=1, max_tool_calls=8, elapsed_seconds=1, max_seconds=30))
        self.assertEqual(research_stop_reason(insufficient, iteration=2, max_iterations=2,
            tool_calls=4, max_tool_calls=8, elapsed_seconds=1, max_seconds=30), "maximum_research_iterations_reached")
        self.assertEqual(research_stop_reason(insufficient, iteration=0, max_iterations=2,
            tool_calls=8, max_tool_calls=8, elapsed_seconds=1, max_seconds=30), "maximum_tool_calls_reached")
        self.assertEqual(research_stop_reason(insufficient, iteration=0, max_iterations=2,
            tool_calls=1, max_tool_calls=8, elapsed_seconds=30, max_seconds=30), "research_time_limit")

    def test_unreviewed_coverage_retries_only_planned_gaps(self):
        plan = ResearchPlan.from_intent("Compare frameworks", [
            {"step": 1, "tool": "web_search", "subquestion": "What architecture do they use?", "instruction": "Search architecture", "depends_on": []},
        ], {"research": {"objective": "Compare frameworks", "subquestions": [
            {"question": "What architecture do they use?", "required_capabilities": ["web_search"]},
            {"question": "What are deployment options?", "required_capabilities": ["web_fetch"]},
        ]}})
        assessment = unreviewed_coverage_assessment(plan, [{
            "subquestion": "What architecture do they use?", "status": "completed",
            "attempts": [{"wrapper": "web_search", "status": "success", "result": "LangGraph nodes and graph architecture"}],
        }], "provider error")
        self.assertFalse(assessment.sufficient)
        self.assertEqual(assessment.unresolved_questions, ["What are deployment options?"])
        self.assertEqual(len(assessment.follow_up_queries), 1)
        self.assertEqual(assessment.follow_up_queries[0].tool, "web_fetch")

    def test_unreviewed_coverage_preserves_explicit_source_conflict(self):
        question = "Does the framework support durable execution after worker restart?"
        plan = ResearchPlan.from_intent("Check durable execution", [
            {"step": 1, "tool": "web_search", "subquestion": question, "instruction": "Search docs", "depends_on": []},
            {"step": 2, "tool": "web_fetch", "subquestion": question, "instruction": "Fetch repo docs", "depends_on": []},
        ], {"research": {"objective": "Check durable execution", "subquestions": [
            {"question": question, "required_capabilities": ["web_search", "web_fetch"]},
        ]}})
        assessment = unreviewed_coverage_assessment(plan, [
            {"evidence_id": 1, "subquestion": question, "status": "completed", "attempts": [
                {"wrapper": "web_search", "status": "success", "result": "The framework supports durable execution after restart."},
            ]},
            {"evidence_id": 2, "subquestion": question, "status": "completed", "attempts": [
                {"wrapper": "web_fetch", "status": "success", "result": "The framework does not support durable execution after restart."},
            ]},
        ], "reviewer unavailable")
        self.assertFalse(assessment.sufficient)
        self.assertEqual(len(assessment.conflicts), 1)
        self.assertEqual(assessment.conflicts[0].evidence_refs, [1, 2])
        self.assertFalse(assessment.conflicts[0].resolved)

    def test_unreviewed_coverage_does_not_invent_gap_when_each_question_has_successful_evidence(self):
        plan = ResearchPlan.from_intent("Compare A and B", [
            {"step": 1, "tool": "web_search", "instruction": "Search A's architecture", "depends_on": []},
            {"step": 2, "tool": "web_fetch", "instruction": "Fetch B's deployment documentation", "depends_on": []},
        ], {"research": {"objective": "Compare", "subquestions": [
            {"question": "What architecture does A use?", "required_capabilities": ["web_search"]},
            {"question": "What deployment options does B document?", "required_capabilities": ["web_fetch"]},
        ]}})
        self.assertEqual([s.subquestion for s in plan.research_steps], [
            "What architecture does A use?", "What deployment options does B document?",
        ])
        assessment = unreviewed_coverage_assessment(plan, [
            {"subquestion": plan.subquestions[0], "status": "completed", "attempts": [
                {"wrapper": "web_search", "status": "success", "result": "Architecture evidence"}]},
            {"subquestion": plan.subquestions[1], "status": "completed", "attempts": [
                {"wrapper": "web_fetch", "status": "success", "result": "Deployment evidence"}]},
        ], "reviewer unavailable")
        self.assertEqual(assessment.unresolved_questions, [])
        self.assertEqual(assessment.follow_up_queries, [])

    def test_unreviewed_coverage_does_not_repeat_question_with_successful_evidence(self):
        question = "What are deployment options for the framework?"
        plan = ResearchPlan.from_intent("Research deployment", [
            {"step": 1, "tool": "web_search", "subquestion": question, "instruction": "Search deployment options", "depends_on": []},
        ], {"research": {"objective": "Research", "subquestions": [
            {"question": question, "required_capabilities": ["web_search", "web_fetch"]},
        ]}})
        assessment = unreviewed_coverage_assessment(plan, [{
            "subquestion": question, "status": "completed", "attempts": [
                {"wrapper": "web_search", "status": "success", "result": "Some deployment details"}],
        }], "reviewer unavailable")
        self.assertEqual(assessment.unresolved_questions, [])
        self.assertEqual(assessment.follow_up_queries, [])


class ResearchReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_reviewer_bounds_large_evidence_but_keeps_provenance(self):
        payload = {"sufficient": False, "addressed_subquestions": [1], "unresolved_questions": [],
                   "conflicts": [], "follow_up_queries": [], "stopping_reason": "Need review"}
        call = AsyncMock(return_value=json.dumps(payload))
        plan = ResearchPlan.from_intent("Review", [
            {"step": 1, "tool": "web_fetch", "instruction": "Fetch docs", "depends_on": []},
        ], {})
        evidence = [{"evidence_id": 1, "subquestion": "Review docs", "source_type": "official_docs",
                    "url": "https://example.com/docs", "timestamp": "2026-01-01", "status": "completed",
                    "attempts": [{"wrapper": "web_fetch", "status": "success", "url": "https://example.com/docs",
                                  "result": "x" * 10000}]}]
        with patch("app.orchestrator.research.call_openai_with_system", new=call):
            await ResearchSufficiencyReviewer().assess(plan, evidence)
        prompt = call.await_args.kwargs["user_prompt"]
        self.assertEqual(call.await_args.kwargs["max_tokens"], 3000)
        self.assertEqual(call.await_args.kwargs["reasoning_effort"], "low")
        self.assertIn("https://example.com/docs", prompt)
        self.assertIn("official_docs", prompt)
        self.assertLess(len(prompt), 5000)

    async def test_reviewer_bounds_total_evidence_across_many_long_attempts(self):
        payload = {"sufficient": False, "addressed_subquestions": [], "unresolved_questions": ["Need more evidence"],
                   "conflicts": [], "follow_up_queries": [], "stopping_reason": "Incomplete evidence"}
        call = AsyncMock(return_value=json.dumps(payload))
        plan = ResearchPlan.from_intent("Compare products", [], {})
        plan.subquestions = ["Compare product claims"]
        evidence = [{"evidence_id": i, "subquestion": "Compare product claims", "status": "completed",
                     "attempts": [{"wrapper": "web_search", "status": "success", "result": "x" * 3000} for _ in range(3)]}
                    for i in range(8)]
        with patch("app.orchestrator.research.call_openai_with_system", new=call):
            await ResearchSufficiencyReviewer().assess(plan, evidence)
        prompt = call.await_args.kwargs["user_prompt"]
        self.assertLess(len(prompt), 15000)
        self.assertEqual(prompt.count("result_excerpt"), 6)

    async def test_reviewer_flags_unresolved_gap_and_requests_new_search(self):
        payload = {
            "sufficient": False,
            "addressed_subquestions": [1],
            "unresolved_questions": ["Deployment models differ"],
            "conflicts": [],
            "follow_up_queries": [{
                "subquestion": "How does deployment compare?",
                "tool": "web_search",
                "instruction": "Find official deployment documentation for framework A and framework B",
            }],
            "stopping_reason": "Deployment detail is missing",
        }
        plan = ResearchPlan.from_intent("Compare A and B", [
            {"step": 1, "tool": "web_search", "instruction": "Compare framework features", "depends_on": []},
        ], {})
        with patch("app.orchestrator.research.call_openai_with_system", new=AsyncMock(return_value=json.dumps(payload))):
            result = await ResearchSufficiencyReviewer().assess(plan, [{"evidence_id": 1, "status": "success"}])
        self.assertFalse(result.sufficient)
        self.assertEqual(len(result.follow_up_queries), 1)

    async def test_reviewer_retries_malformed_output_and_then_returns_assessment(self):
        payload = {"sufficient": True, "addressed_subquestions": [1], "unresolved_questions": [],
                   "conflicts": [], "follow_up_queries": [], "stopping_reason": "Enough relevant evidence"}
        call = AsyncMock(side_effect=["not json", json.dumps(payload)])
        plan = ResearchPlan.from_intent("Find a paper", [
            {"step": 1, "tool": "semantic_scholar_search", "instruction": "Search for paper", "depends_on": []},
        ], {})
        with patch("app.orchestrator.research.call_openai_with_system", new=call):
            result = await ResearchSufficiencyReviewer().assess(plan, [{"evidence_id": 1}])
        self.assertTrue(result.sufficient)
        self.assertEqual(call.await_count, 2)

    async def test_reviewer_provider_failure_returns_limitation_instead_of_raising(self):
        plan = ResearchPlan.from_intent("Find papers", [
            {"step": 1, "tool": "semantic_scholar_search", "instruction": "Search for papers", "depends_on": []},
        ], {})
        with patch("app.orchestrator.research.call_openai_with_system", new=AsyncMock(side_effect=RuntimeError("provider unavailable"))):
            result = await ResearchSufficiencyReviewer().assess(plan, [{"evidence_id": 1}])
        self.assertFalse(result.sufficient)
        self.assertIn("Evidence reviewer unavailable", result.stopping_reason)
        self.assertEqual(len(result.follow_up_queries), 1)


class PlannerResearchValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_dependency_is_rejected_then_retried(self):
        valid = {"mode": "tool_execution", "answer": "", "research": None, "steps": [
            {"tool": "web_search", "instruction": "Search recent AI agent developments", "depends_on": []}
        ]}
        call = AsyncMock(side_effect=[
            json.dumps({"mode": "tool_execution", "answer": "", "steps": [
                {"tool": "web_search", "instruction": "Search", "depends_on": [1]}
            ]}),
            json.dumps(valid),
        ])
        with patch("app.agents.planner.call_openai_with_system", new=call):
            result = await PlannerAgent().plan_intent("Search recent AI agent developments")
        self.assertEqual(call.await_count, 2)
        self.assertEqual(result["steps"][0]["tool"], "web_search")


if __name__ == "__main__":
    unittest.main()
