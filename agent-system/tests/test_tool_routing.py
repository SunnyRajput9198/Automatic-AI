import json
import unittest
from unittest.mock import AsyncMock, patch

from app.agents.executor import ExecutorAgent
from app.agents.planner import PlannerAgent
from app.orchestrator.loop_v3 import _synthesis_has_grounded_links
from app.tools.base import Tool, ToolResult


class FakeTool(Tool):
    def __init__(self, tool_name, result=None):
        self.tool_name = tool_name
        self.calls = []
        self.result = result or ToolResult(
            success=True, output=f"{tool_name} result", metadata={"tool_name": tool_name}
        )

    @property
    def name(self):
        return self.tool_name

    @property
    def description(self):
        return "Test wrapper"

    @property
    def input_schema(self):
        return {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class PlannerIntentTests(unittest.IsolatedAsyncioTestCase):
    async def make_plan(self, payload, user_task="test request"):
        with patch("app.agents.planner.call_openai_with_system", new=AsyncMock(return_value=json.dumps(payload))):
            return await PlannerAgent().plan_intent(user_task)

    async def test_direct_response_has_no_executable_steps(self):
        plan = await self.make_plan({"mode": "direct_response", "answer": "Hello!", "steps": []})
        self.assertEqual(plan["mode"], "direct_response")
        self.assertEqual(plan["steps"], [])

    async def test_internal_json_answer_is_normalized_to_prose(self):
        plan = await self.make_plan({"mode": "direct_response", "answer": '{"response":"Hello!","intent":{"name":"greet"}}', "steps": []})
        self.assertEqual(plan["answer"], "Hello!")

    def test_prose_prefixed_internal_intent_is_removed(self):
        self.assertEqual(
            PlannerAgent._normalise_direct_answer('Hello! Intent JSON: intent = "greeting", confidence = 1.0'),
            "Hello!",
        )

    async def test_dynamic_weather_request_cannot_fall_through_as_direct_answer(self):
        plan = await self.make_plan({
            "mode": "direct_response", "answer": "Slots: city Delhi, date today", "steps": [],
        }, "What is the weather in Delhi today?")
        self.assertEqual(plan["mode"], "tool_execution")
        self.assertEqual(plan["steps"][0]["tool"], "get_weather")

    async def test_academic_request_uses_scholarly_source_even_if_planner_picks_web(self):
        plan = await self.make_plan({
            "mode": "tool_execution", "answer": "", "steps": [
                {"tool": "web_search", "instruction": "Search the web about RAG", "depends_on": []}
            ],
        }, "Find recent academic research about RAG")
        self.assertEqual(plan["steps"][0]["tool"], "semantic_scholar_search")
        self.assertIsNotNone(plan["research"])

    async def test_academic_capability_overrides_contradictory_generic_research_capability(self):
        plan = await self.make_plan({
            "mode": "tool_execution", "answer": "", "research": {
                "objective": "Find academic research about RAG",
                "subquestions": [{"question": "Find academic research", "required_capabilities": ["web_search"]}],
            },
            "steps": [{"tool": "web_search", "instruction": "Search RAG papers", "depends_on": []}],
        }, "Find recent academic research about RAG")
        self.assertEqual(plan["steps"][0]["tool"], "semantic_scholar_search")

    def test_synthesis_rejects_links_absent_from_successful_evidence(self):
        evidence = [{"attempts": [{"status": "success", "source_url": "https://example.com/a", "result": "Found https://example.com/b."}]}]
        self.assertTrue(_synthesis_has_grounded_links("https://example.com/a https://example.com/b", evidence))
        self.assertFalse(_synthesis_has_grounded_links("https://example.com/a https://invented.example/x", evidence))

    async def test_malformed_plan_is_retried_once_and_repaired(self):
        call = AsyncMock(side_effect=[
            '{"mode":"tool_execution","answer":"","steps":[{"tool":"web_search"}]}',
            '{"mode":"tool_execution","answer":"","steps":[{"tool":"web_search","instruction":"Search current AI agent developments","depends_on":[]}]}',
        ])
        with patch("app.agents.planner.call_openai_with_system", new=call):
            result = await PlannerAgent().plan_intent("Search for recent AI agents research")
        self.assertEqual(result["steps"][0]["tool"], "web_search")
        self.assertEqual(call.await_count, 2)

    async def test_search_request_selects_web_search(self):
        plan = await self.make_plan({"mode": "tool_execution", "answer": "", "steps": [
            {"tool": "web_search", "instruction": "Use web_search for the latest AI agents information", "depends_on": []}
        ]})
        self.assertEqual(plan["mode"], "tool_execution")
        self.assertIn("web_search", plan["steps"][0]["instruction"])

    async def test_github_request_selects_github_wrapper(self):
        plan = await self.make_plan({"mode": "tool_execution", "answer": "", "steps": [
            {"tool": "github", "instruction": "Use github to get the latest commit from openai/openai-python", "depends_on": []}
        ]})
        self.assertIn("github", plan["steps"][0]["instruction"])

    async def test_weather_request_selects_weather_wrapper(self):
        plan = await self.make_plan({"mode": "tool_execution", "answer": "", "steps": [
            {"tool": "get_weather", "instruction": "Use get_weather for Delhi", "depends_on": []}
        ]})
        self.assertIn("get_weather", plan["steps"][0]["instruction"])

    async def test_python_request_selects_python_wrapper_when_computation_is_needed(self):
        plan = await self.make_plan({"mode": "tool_execution", "answer": "", "steps": [
            {"tool": "python_executor", "instruction": "Use python_executor to calculate the median of [1, 2, 3, 4, 100]", "depends_on": []}
        ]})
        self.assertIn("python_executor", plan["steps"][0]["instruction"])

    async def test_multi_tool_plan_has_both_capabilities(self):
        plan = await self.make_plan({"mode": "multi_tool", "answer": "", "steps": [
            {"tool": "github", "instruction": "Use github to retrieve latest release from openai/openai-python", "depends_on": []},
            {"tool": "get_weather", "instruction": "Use get_weather for Delhi", "depends_on": []},
        ]})
        self.assertEqual(plan["mode"], "multi_tool")
        self.assertEqual(len(plan["steps"]), 2)


class ExecutorRoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_fallback_policy_requires_compatible_public_source(self):
        can_fallback = ExecutorAgent.can_fallback
        self.assertTrue(can_fallback("github", "web_search", "latest commit from public repo owner/name"))
        self.assertFalse(can_fallback("github", "web_search", "latest commit from my private repo"))
        self.assertFalse(can_fallback("get_weather", "web_search", "weather in Delhi"))

    async def dispatch(self, tool_name, arguments=None, result=None):
        executor = ExecutorAgent()
        fake = FakeTool(tool_name, result)
        executor.tools = {tool_name: fake}
        call = {"type": "tool_call", "name": tool_name, "arguments": arguments or {}}
        with patch("app.agents.executor.call_openai_with_tools", new=AsyncMock(return_value=call)):
            output = await executor.execute_step(
                f"Use {tool_name} for the request", context={"forced_tool": tool_name}
            )
        return output, fake

    async def test_search_routes_to_search_wrapper(self):
        result, tool = await self.dispatch("web_search", {"query": "latest AI agents"})
        self.assertTrue(result.success)
        self.assertEqual(len(tool.calls), 1)

    async def test_github_routes_to_github_wrapper(self):
        result, tool = await self.dispatch("github", {"owner": "openai", "repo": "openai-python", "operation": "latest_commit"})
        self.assertTrue(result.success)
        self.assertEqual(tool.calls[0]["owner"], "openai")

    async def test_weather_routes_to_weather_wrapper(self):
        result, tool = await self.dispatch("get_weather", {"city": "Delhi"})
        self.assertTrue(result.success)
        self.assertEqual(tool.calls[0]["city"], "Delhi")

    async def test_python_routes_to_python_wrapper(self):
        result, tool = await self.dispatch("python_executor", {"code": "print(3)"})
        self.assertTrue(result.success)
        self.assertEqual(len(tool.calls), 1)

    async def test_multi_tool_request_dispatches_both_wrappers(self):
        executor = ExecutorAgent()
        github, weather = FakeTool("github"), FakeTool("get_weather")
        executor.tools = {"github": github, "get_weather": weather}
        tool_call = AsyncMock(side_effect=[
            {"type": "tool_call", "name": "github", "arguments": {"owner": "openai", "repo": "openai-python", "operation": "latest_release"}},
            {"type": "tool_call", "name": "get_weather", "arguments": {"city": "Delhi"}},
        ])
        with patch("app.agents.executor.call_openai_with_tools", new=tool_call):
            first = await executor.execute_step("Use github to get release details", context={"forced_tool": "github"})
            second = await executor.execute_step("Use get_weather for Delhi", context={"forced_tool": "get_weather"})
        self.assertTrue(first.success and second.success)
        self.assertEqual(len(github.calls), 1)
        self.assertEqual(len(weather.calls), 1)

    async def test_public_github_failure_falls_back_to_web_search(self):
        executor = ExecutorAgent()
        github = FakeTool("github", ToolResult(success=False, output="", error="provider unavailable"))
        search = FakeTool("web_search", ToolResult(
            success=True,
            output="Latest commit: abcdef0123456789\nhttps://github.com/o/r/commit/abcdef0123456789",
            metadata={"tool_name": "web_search", "source": ["DuckDuckGo"]},
        ))
        executor.tools = {"github": github, "web_search": search}
        call = AsyncMock(side_effect=[
            {"type": "tool_call", "name": "github", "arguments": {"owner": "o", "repo": "r", "operation": "latest_commit"}},
            {"type": "tool_call", "name": "web_search", "arguments": {"query": "official commit"}},
        ])
        with patch("app.agents.executor.call_openai_with_tools", new=call):
            result, attempts = await executor.execute_with_fallbacks(
                "Get the latest public commit", "github", ["web_search"],
                "Get latest commit from public GitHub repository o/r",
            )
        self.assertTrue(result.success)
        self.assertEqual([name for name, _ in attempts], ["github", "web_search"])
        self.assertEqual(result.metadata["fallback_from"], "github")

    async def test_private_github_failure_does_not_fall_back_to_search(self):
        executor = ExecutorAgent()
        github = FakeTool("github", ToolResult(success=False, output="", error="authentication failed"))
        search = FakeTool("web_search")
        executor.tools = {"github": github, "web_search": search}
        call = AsyncMock(return_value={
            "type": "tool_call", "name": "github",
            "arguments": {"owner": "o", "repo": "r", "operation": "latest_commit"},
        })
        with patch("app.agents.executor.call_openai_with_tools", new=call):
            result, attempts = await executor.execute_with_fallbacks(
                "Get the latest commit", "github", ["web_search"],
                "Get the latest commit from my private repository o/r",
            )
        self.assertFalse(result.success)
        self.assertEqual([name for name, _ in attempts], ["github"])
        self.assertEqual(search.calls, [])

    async def test_github_fallback_rejects_unverified_search_results(self):
        executor = ExecutorAgent()
        github = FakeTool("github", ToolResult(success=False, output="", error="provider unavailable"))
        search = FakeTool("web_search", ToolResult(success=True, output="Unrelated search result"))
        executor.tools = {"github": github, "web_search": search}
        call = AsyncMock(side_effect=[
            {"type": "tool_call", "name": "github", "arguments": {"owner": "o", "repo": "r", "operation": "latest_commit"}},
            {"type": "tool_call", "name": "web_search", "arguments": {"query": "public repo commit"}},
        ])
        with patch("app.agents.executor.call_openai_with_tools", new=call):
            result, attempts = await executor.execute_with_fallbacks(
                "Find latest commit", "github", ["web_search"],
                "Get latest commit from public GitHub repository o/r",
            )
        self.assertFalse(result.success)
        self.assertEqual(result.metadata["failure_type"], "UNVERIFIED_FALLBACK")
        self.assertEqual(attempts[1][1].metadata["validation_status"], "unverified")

    async def test_wrapper_failure_is_returned_without_crashing(self):
        failed = ToolResult(success=False, output="", error="upstream unavailable")
        result, tool = await self.dispatch("web_search", {"query": "latest AI"}, failed)
        self.assertFalse(result.success)
        self.assertEqual(result.error, "upstream unavailable")
        self.assertEqual(len(tool.calls), 1)


if __name__ == "__main__":
    unittest.main()
