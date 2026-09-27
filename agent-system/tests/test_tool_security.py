import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from app.tools.shell_tool import ShellExecutor
from app.tools.base import ToolResult, classify_tool_failure
from app.tools.web_search import WebFetchTool, WebSearchTool
from app.tools.weather_tool import WeatherTool
from app.utils.file_manager import FileManager


class WrapperSecurityTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_fetch_rejects_local_and_private_targets(self):
        tool = WebFetchTool()
        for url in ("http://example.com", "https://localhost/", "https://127.0.0.1/", "https://[::1]/", "https://user:pass@example.com/"):
            with self.subTest(url=url):
                result = await tool.run(url=url)
                self.assertFalse(result.success)

    def test_shell_rejects_chaining_and_traversal(self):
        tool = ShellExecutor()
        self.assertFalse(tool._is_command_safe("cat file.txt && whoami"))
        self.assertFalse(tool._is_command_safe("cat ../../secret.txt"))
        self.assertFalse(tool._is_command_safe("find . -exec whoami \\\";\\\""))
        self.assertTrue(tool._is_command_safe("cat notes.txt"))

    def test_task_workspaces_reject_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = FileManager(directory)
            with self.assertRaises(ValueError):
                manager.get_task_workspace("../outside")
            self.assertFalse(manager.cleanup_task_workspace("../outside"))

    async def test_web_fetch_revalidates_redirect_destination(self):
        tool = WebFetchTool()
        with patch.object(tool, "_validate_public_url", new=AsyncMock(side_effect=[None, ValueError("private destination")])):
            with patch("app.tools.web_search.httpx.AsyncClient") as client_cls:
                client = client_cls.return_value.__aenter__.return_value
                client.get = AsyncMock(return_value=type("Response", (), {"status_code": 302, "headers": {"location": "https://127.0.0.1/"}})())
                result = await tool.run(url="https://example.com/start")
        self.assertFalse(result.success)
        self.assertIn("private destination", result.error)

    async def test_weather_network_failure_uses_separately_attributed_fallback(self):
        tool = WeatherTool()
        fallback = ToolResult(success=True, output="Current weather from fallback", metadata={
            "tool_name": "get_weather", "source": "wttr.in", "fallback_from": "Open-Meteo"
        })
        with patch.object(tool, "_get_retrying", new=AsyncMock(side_effect=__import__("httpx").ConnectTimeout("timeout"))):
            with patch.object(tool, "_fallback_wttr", new=AsyncMock(return_value=fallback)) as fallback_call:
                result = await tool.run(city="Delhi")
        self.assertTrue(result.success)
        self.assertEqual(result.metadata["source"], "wttr.in")
        fallback_call.assert_awaited_once()

    def test_upstream_failure_types_are_distinguished(self):
        self.assertEqual(classify_tool_failure("timeout", 504), "TIMEOUT")
        self.assertEqual(classify_tool_failure("rate limit", 429), "RATE_LIMIT")
        self.assertEqual(classify_tool_failure("unauthorized", 401), "AUTHENTICATION_FAILURE")
        self.assertEqual(classify_tool_failure("connection refused"), "NETWORK_ERROR")
        self.assertEqual(classify_tool_failure("No results found"), "EMPTY_RESULT")

    async def test_search_reformulates_when_primary_sources_return_no_results(self):
        tool = WebSearchTool()
        result_item = {"source": "DuckDuckGo", "title": "Agent research", "content": "Recent results", "url": "https://example.org"}
        with patch.object(tool._wiki, "search", new=AsyncMock(return_value=[])):
            with patch.object(tool, "_ddg_search", new=AsyncMock(side_effect=[[], [result_item]])) as search:
                result = await tool.run(query="AI agents", max_results=5)
        self.assertTrue(result.success)
        self.assertEqual(search.await_count, 2)
        self.assertIn("recent information sources", search.await_args.args[0])


if __name__ == "__main__":
    unittest.main()
