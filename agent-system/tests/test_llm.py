import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.utils import llm


class LLMWrapperTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_wrapper_forwards_model_to_sync_client(self):
        with (
            patch.object(llm.openai_rate_limiter, "wait_if_needed", new_callable=AsyncMock),
            patch("app.utils.llm._call_with_retries", new_callable=AsyncMock) as run,
        ):
            run.return_value = "OK"

            result = await llm.call_openai(
                [{"role": "user", "content": "Say OK"}], model="gpt-5-mini"
            )

        self.assertEqual(result, "OK")
        self.assertEqual(run.await_args.args, (llm._sync_openai_call,))
        self.assertEqual(run.await_args.kwargs["model"], "gpt-5-mini")
        self.assertEqual(run.await_args.kwargs["telemetry_model"], "gpt-5-mini")
        self.assertNotIn("model_name", run.await_args.kwargs)

    async def test_retry_runner_passes_model_and_other_arguments_to_client(self):
        calls = []

        def client(*, model, prompt):
            calls.append((model, prompt))
            return "OK"

        result = await llm._call_with_retries(
            client,
            model="gpt-5-mini",
            prompt="Say OK",
            provider="ai_credits",
            telemetry_model="gpt-5-mini",
        )

        self.assertEqual(result, "OK")
        self.assertEqual(calls, [("gpt-5-mini", "Say OK")])

    async def test_attempt_capture_saves_sanitized_request_and_response_metadata(self):
        def client(*, messages, model, max_tokens, temperature, reasoning_effort):
            return "OK"

        with llm.capture_llm_attempts() as attempts:
            result = await llm._call_with_retries(
                client,
                messages=[{"role": "user", "content": "Say OK"}],
                model="gpt-5-mini", max_tokens=3000, temperature=0.1,
                reasoning_effort="low", provider="ai_credits", telemetry_model="gpt-5-mini",
            )

        self.assertEqual(result, "OK")
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "success")
        self.assertEqual(attempts[0]["input_characters"], 6)
        self.assertEqual(attempts[0]["reasoning_effort"], "low")
        self.assertNotIn("messages", attempts[0])

    async def test_attempt_capture_saves_failure_without_prompt_or_credentials(self):
        def client(*, messages, model, max_tokens):
            raise ValueError("Empty response from OpenAI via LangChain")

        with llm.capture_llm_attempts() as attempts:
            with self.assertRaises(RuntimeError):
                await llm._call_with_retries(
                    client, messages=[{"role": "user", "content": "private prompt"}],
                    model="gpt-5-mini", max_tokens=1400, max_retries=1,
                    provider="ai_credits", telemetry_model="gpt-5-mini",
                )

        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["status"], "error")
        self.assertEqual(attempts[0]["error_category"], "EMPTY_RESPONSE_ERROR")
        self.assertEqual(attempts[0]["input_characters"], len("private prompt"))
        self.assertNotIn("private prompt", str(attempts))

    def test_openai_client_receives_low_reasoning_effort_when_requested(self):
        response = MagicMock()
        response.content = "{}"
        response.response_metadata = {}
        response.usage_metadata = {}
        response.additional_kwargs = {}
        response.tool_calls = []
        client = MagicMock()
        client.invoke.return_value = response
        with (
            patch("app.utils.llm._get_openai_api_key", return_value="test-key"),
            patch("app.utils.llm._get_openai_base_url", return_value="https://aicredits.in/v1"),
            patch("app.utils.llm.ChatOpenAI", return_value=client) as chat_openai,
        ):
            llm._sync_openai_call([{"role": "user", "content": "Review"}], "gpt-5-mini", 0.1, 3000, "low")

        self.assertEqual(chat_openai.call_args.kwargs["reasoning_effort"], "low")

    async def test_retry_runner_retries_transient_failures(self):
        calls = 0

        def client():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("request timed out")
            return "OK"

        with patch("app.utils.llm.asyncio.sleep", new_callable=AsyncMock):
            result = await llm._call_with_retries(
                client, max_retries=2, initial_delay=0
            )

        self.assertEqual(result, "OK")
        self.assertEqual(calls, 2)

    def test_langchain_openai_client_uses_ai_credits_configuration(self):
        response = MagicMock()
        response.content = "OK"
        client = MagicMock()
        client.invoke.return_value = response

        with (
            patch("app.utils.llm._get_openai_api_key", return_value="test-key"),
            patch(
                "app.utils.llm._get_openai_base_url",
                return_value="https://aicredits.in/v1",
            ),
            patch("app.utils.llm.ChatOpenAI", return_value=client) as chat_openai,
        ):
            result = llm._sync_openai_call(
                [{"role": "user", "content": "Say OK"}],
                "gpt-5-mini",
                0,
                20,
            )

        self.assertEqual(result, "OK")
        self.assertEqual(chat_openai.call_args.kwargs["base_url"], "https://aicredits.in/v1")
        self.assertEqual(chat_openai.call_args.kwargs["model"], "gpt-5-mini")

    def test_empty_response_logs_sanitized_finish_and_usage_metadata(self):
        response = MagicMock()
        response.content = ""
        response.response_metadata = {
            "finish_reason": "length", "model_name": "gpt-5-mini",
            "token_usage": {"prompt_tokens": 20, "completion_tokens": 1400, "total_tokens": 1420},
        }
        response.usage_metadata = {"input_tokens": 20, "output_tokens": 1400, "total_tokens": 1420}
        response.additional_kwargs = {}
        response.tool_calls = []
        client = MagicMock()
        client.invoke.return_value = response
        with (
            patch("app.utils.llm._get_openai_api_key", return_value="test-key"),
            patch("app.utils.llm._get_openai_base_url", return_value="https://aicredits.in/v1"),
            patch("app.utils.llm.ChatOpenAI", return_value=client),
            patch("app.utils.llm.logger") as logger,
        ):
            with self.assertRaisesRegex(ValueError, "Empty response"):
                llm._sync_openai_call([{"role": "user", "content": "test"}], "gpt-5-mini", 0.1, 1400)

        diagnostic = logger.error.call_args.kwargs
        self.assertEqual(diagnostic["content_characters"], 0)
        self.assertEqual(diagnostic["finish_reason"], "length")
        self.assertEqual(diagnostic["provider_token_usage"]["completion_tokens"], 1400)
        self.assertNotIn("api_key", diagnostic)

    def test_empty_langchain_response_has_distinct_retryable_classification(self):
        diagnostic = llm.classify_llm_exception(ValueError("Empty response from OpenAI via LangChain"))
        self.assertEqual(diagnostic["category"], "EMPTY_RESPONSE_ERROR")
        self.assertTrue(diagnostic["retryable"])

    def test_classifies_wrapped_provider_error(self):
        cause = RuntimeError("401 invalid_api_key")
        wrapped = RuntimeError("request failed")
        wrapped.__cause__ = cause

        diagnostic = llm.classify_llm_exception(wrapped)

        self.assertEqual(diagnostic["category"], "AUTHENTICATION_ERROR")
        self.assertFalse(diagnostic["retryable"])

    def test_sanitizes_api_keys(self):
        self.assertEqual(
            llm.sanitize_error("request failed with sk-ant-api03-12345678901234567890"),
            "request failed with sk-***",
        )


if __name__ == "__main__":
    unittest.main()
