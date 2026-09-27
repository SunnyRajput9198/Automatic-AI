import unittest

from app.tools.python_tool import RestrictedPythonExecutor


class RestrictedPythonExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_print_output_is_captured(self):
        result = await RestrictedPythonExecutor().run(code="print('Hello.')")

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output, "Hello.\n")

    async def test_code_without_print_returns_empty_output(self):
        result = await RestrictedPythonExecutor().run(code="value = 1 + 1")

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output, "")

    async def test_safe_statistics_import_is_available(self):
        result = await RestrictedPythonExecutor().run(
            code="import statistics\nprint(statistics.median([1, 3, 5]))"
        )

        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output, "3\n")

    async def test_unsafe_import_is_blocked(self):
        result = await RestrictedPythonExecutor().run(code="import os\nprint(os.name)")

        self.assertFalse(result.success)
        self.assertIn("Only math and statistics", result.error)
