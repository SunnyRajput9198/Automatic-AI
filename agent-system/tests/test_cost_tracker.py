import json
import tempfile
import unittest
from unittest.mock import patch

from app.utils.cost_tracker import CostTracker


class CostTrackerDiagnosticTests(unittest.TestCase):
    def test_saves_sanitized_llm_attempt_diagnostics_with_task_cost(self):
        with tempfile.TemporaryDirectory() as costs_dir, patch("app.utils.cost_tracker.COSTS_DIR", costs_dir):
            tracker = CostTracker()
            tracker.start_task("review-test")
            tracker.record_llm_diagnostic({
                "provider": "ai_credits", "model": "gpt-5-mini", "attempt": 1,
                "status": "error", "finish_reason": "length", "input_characters": 1200,
                "max_tokens": 3000, "provider_token_usage": {"completion_tokens": 3000},
            })
            tracker.complete_task(success=True)

            with open(f"{costs_dir}/task_review-test.json", encoding="utf-8") as cost_file:
                data = json.load(cost_file)

        diagnostic = data["llm_call_diagnostics"][0]
        self.assertEqual(diagnostic["finish_reason"], "length")
        self.assertEqual(diagnostic["provider_token_usage"]["completion_tokens"], 3000)


if __name__ == "__main__":
    unittest.main()
