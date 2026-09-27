import unittest

from app.utils.json_parser import extract_json


class ExtractJsonTests(unittest.TestCase):
    def test_extracts_object_with_braces_inside_string_values(self):
        payload = {"sufficient": False, "stopping_reason": "The docs contain {deploy} but not a self-hosted section."}
        import json
        response = "Decision:\n" + json.dumps(payload) + "\nReview complete."
        self.assertEqual(extract_json(response, context="review"), payload)

    def test_extracts_object_from_markdown_fence(self):
        payload = {"sufficient": True, "addressed_subquestions": [1]}
        import json
        response = "```json\n" + json.dumps(payload) + "\n```"
        self.assertEqual(extract_json(response, context="review"), payload)


if __name__ == "__main__":
    unittest.main()
