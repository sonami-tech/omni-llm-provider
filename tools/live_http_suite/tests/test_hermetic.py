#!/usr/bin/env python3
"""Hermetic unit tests for live HTTP suite oracles, SSE parse, and retry policy.

No network. Run:
  python3 -m unittest tools.live_http_suite.tests.test_hermetic -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.live_http_suite.client import should_retry_transport  # noqa: E402
from tools.live_http_suite.config import (  # noqa: E402
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_DUAL_MODE_MODEL,
    SuiteConfig,
)
from tools.live_http_suite import oracles as O  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_dual_mode_off_from_env_before_pin(self) -> None:
        # WHY: skip flag must be readable before dual-mode pin is resolved so a
        # missing Grok catalog entry never becomes an inferred post-4xx skip.
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"OMNI_TEST_DUAL_MODE_OFF": "1"}, clear=False):
            cfg = SuiteConfig.from_env()
            self.assertTrue(cfg.dual_mode_off)
            # Pin is still populated (for when flag is cleared later); skip is separate.
            self.assertEqual(cfg.dual_mode_model, DEFAULT_DUAL_MODE_MODEL)

    def test_cli_dual_mode_off_overrides_env_false(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"OMNI_TEST_DUAL_MODE_OFF": "0"}, clear=False):
            cfg = SuiteConfig.from_env(dual_mode_off=True)
            self.assertTrue(cfg.dual_mode_off)

    def test_responses_model_defaults_to_claude_pin(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(
            os.environ,
            {
                "OMNI_TEST_CLAUDE_MODEL": "claude-haiku-4-5-20251001",
            },
            clear=False,
        ):
            # Clear responses override if present
            env = dict(os.environ)
            env.pop("OMNI_TEST_RESPONSES_MODEL", None)
            with mock.patch.dict(os.environ, env, clear=True):
                cfg = SuiteConfig.from_env()
                self.assertEqual(cfg.responses_model, "claude-haiku-4-5-20251001")


class RetryPolicyTests(unittest.TestCase):
    def test_retry_on_connection_before_body(self) -> None:
        self.assertTrue(
            should_retry_transport(
                status=None, connection_error=True, body_bytes_received=False
            )
        )

    def test_no_retry_after_body_bytes(self) -> None:
        # WHY: once stream/body bytes start, retry would hide partial delivery
        # and can duplicate side effects. Semantic failures also must not retry.
        self.assertFalse(
            should_retry_transport(
                status=503, connection_error=False, body_bytes_received=True
            )
        )
        self.assertFalse(
            should_retry_transport(
                status=None, connection_error=True, body_bytes_received=True
            )
        )

    def test_retry_429_and_5xx_before_body(self) -> None:
        self.assertTrue(
            should_retry_transport(
                status=429, connection_error=False, body_bytes_received=False
            )
        )
        self.assertTrue(
            should_retry_transport(
                status=502, connection_error=False, body_bytes_received=False
            )
        )

    def test_no_retry_on_4xx_client_errors(self) -> None:
        self.assertFalse(
            should_retry_transport(
                status=400, connection_error=False, body_bytes_received=False
            )
        )
        self.assertFalse(
            should_retry_transport(
                status=404, connection_error=False, body_bytes_received=False
            )
        )


class SseParseTests(unittest.TestCase):
    def test_parse_named_events_and_done(self) -> None:
        raw = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"hi"}\n'
            "\n"
            "event: response.completed\n"
            'data: {"type":"response.completed"}\n'
            "\n"
            "data: [DONE]\n"
            "\n"
        )
        events = O.parse_sse(raw)
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].event, "response.output_text.delta")
        self.assertIn("hi", events[0].data)
        self.assertEqual(events[2].data, "[DONE]")

    def test_multiline_data_joined(self) -> None:
        raw = "data: {\"a\":1}\ndata: line2\n\n"
        events = O.parse_sse(raw)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, '{"a":1}\nline2')


class OpenAiErrorOracleTests(unittest.TestCase):
    def test_unknown_model_nonstream_pass(self) -> None:
        body = json.dumps(
            {"error": {"message": "unknown model", "type": "invalid_request_error", "code": None}}
        )
        r = O.oracle_unknown_model_nonstream(
            400, {"Content-Type": "application/json"}, body
        )
        self.assertTrue(r.ok, r.reason)

    def test_unknown_model_rejects_auth_as_false_green(self) -> None:
        # WHY: 401/403/429 must not be counted as unknown-model success.
        body = json.dumps({"error": {"message": "nope", "type": "authentication_error"}})
        for status in (401, 403, 429):
            r = O.oracle_unknown_model_nonstream(
                status, {"Content-Type": "application/json"}, body
            )
            self.assertFalse(r.ok, f"status {status} must fail")
            self.assertIn("false green", r.reason)

    def test_unknown_model_rejects_5xx(self) -> None:
        r = O.oracle_unknown_model_nonstream(
            500, {"Content-Type": "application/json"}, '{"error":{"message":"x"}}'
        )
        self.assertFalse(r.ok)


class StreamXErrorOracleTests(unittest.TestCase):
    def test_path_a_json_error(self) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": "unknown model 'no-such-model-xyz'",
                    "type": "invalid_request_error",
                }
            }
        )
        r = O.oracle_stream_x_error(404, {"Content-Type": "application/json"}, body)
        self.assertTrue(r.ok, r.reason)

    def test_path_b_error_frame(self) -> None:
        raw = (
            'data: {"error":{"message":"bad model","type":"invalid_request_error"}}\n'
            "\n"
            "data: [DONE]\n"
            "\n"
        )
        r = O.oracle_stream_x_error(200, {"Content-Type": "text/event-stream"}, raw)
        self.assertTrue(r.ok, r.reason)

    def test_empty_stream_fails(self) -> None:
        r = O.oracle_stream_x_error(200, {"Content-Type": "text/event-stream"}, "")
        self.assertFalse(r.ok)
        self.assertIn("empty", r.reason.lower())

    def test_normal_success_stream_fails(self) -> None:
        # A happy chat stream with stop + content must not pass as stream×error.
        raw = (
            'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        r = O.oracle_stream_x_error(200, {"Content-Type": "text/event-stream"}, raw)
        self.assertFalse(r.ok, r.reason)

    def test_mixed_error_and_success_fails(self) -> None:
        # WHY: error key must not suppress rejection of normal stop+content.
        raw = (
            'data: {"error":null,"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        r = O.oracle_stream_x_error(200, {"Content-Type": "text/event-stream"}, raw)
        self.assertFalse(r.ok, r.reason)

    def test_error_null_alone_not_error_signal(self) -> None:
        raw = (
            'data: {"error":null,"choices":[{"delta":{},"finish_reason":null}]}\n\n'
            "data: [DONE]\n\n"
        )
        r = O.oracle_stream_x_error(200, {"Content-Type": "text/event-stream"}, raw)
        self.assertFalse(r.ok, r.reason)

    def test_auth_status_fails(self) -> None:
        r = O.oracle_stream_x_error(
            401, {"Content-Type": "application/json"}, '{"error":{"message":"x"}}'
        )
        self.assertFalse(r.ok)


class DroppedToolChoiceOracleTests(unittest.TestCase):
    def test_token_in_merged_value_passes(self) -> None:
        r = O.oracle_dropped_tool_choice({"X-Omni-Dropped": "max_tokens, tool_choice"})
        self.assertTrue(r.ok, r.reason)

    def test_missing_header_fails(self) -> None:
        r = O.oracle_dropped_tool_choice({"Content-Type": "application/json"})
        self.assertFalse(r.ok)

    def test_other_token_only_fails(self) -> None:
        r = O.oracle_dropped_tool_choice({"x-omni-dropped": "max_tokens"})
        self.assertFalse(r.ok)
        self.assertIn("tool_choice", r.reason)


class StreamForcedToolOracleTests(unittest.TestCase):
    def test_assembles_name_args_and_terminal(self) -> None:
        raw = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function",'
            '"function":{"name":"get_weather","arguments":"{\\"city\\""}}]},'
            '"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":":\\"Paris\\"}"}}]},"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )
        r = O.oracle_stream_forced_tool(
            200,
            {"Content-Type": "text/event-stream"},
            raw,
            expected_name="get_weather",
        )
        self.assertTrue(r.ok, r.reason)

    def test_bad_args_json_fails(self) -> None:
        raw = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function",'
            '"function":{"name":"get_weather","arguments":"{not-json"}}]},'
            '"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )
        r = O.oracle_stream_forced_tool(
            200,
            {"Content-Type": "text/event-stream"},
            raw,
            expected_name="get_weather",
        )
        self.assertFalse(r.ok)
        self.assertIn("JSON", r.reason)

    def test_array_args_fails(self) -> None:
        raw = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function",'
            '"function":{"name":"get_weather","arguments":"[1,2]"}}]},'
            '"finish_reason":null}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )
        r = O.oracle_stream_forced_tool(
            200,
            {"Content-Type": "text/event-stream"},
            raw,
            expected_name="get_weather",
        )
        self.assertFalse(r.ok)
        self.assertIn("object", r.reason)

    def test_error_frame_fails(self) -> None:
        raw = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function",'
            '"function":{"name":"get_weather","arguments":"{\\"city\\":\\"x\\"}"}}]},'
            '"finish_reason":null}]}\n\n'
            'data: {"error":{"message":"boom"}}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )
        r = O.oracle_stream_forced_tool(
            200,
            {"Content-Type": "text/event-stream"},
            raw,
            expected_name="get_weather",
        )
        self.assertFalse(r.ok)
        self.assertIn("error", r.reason.lower())


class ResponsesStreamOracleTests(unittest.TestCase):
    def test_happy_text_and_terminal(self) -> None:
        raw = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"pong"}\n\n'
            "event: response.completed\n"
            'data: {"type":"response.completed"}\n\n'
        )
        r = O.oracle_responses_stream_happy(
            200, {"Content-Type": "text/event-stream"}, raw
        )
        self.assertTrue(r.ok, r.reason)

    def test_metadata_only_empty_text_fails(self) -> None:
        # WHY: lifecycle frames alone are a false green for "happy stream".
        raw = (
            "event: response.created\n"
            'data: {"type":"response.created"}\n\n'
            "event: response.completed\n"
            'data: {"type":"response.completed"}\n\n'
        )
        r = O.oracle_responses_stream_happy(
            200, {"Content-Type": "text/event-stream"}, raw
        )
        self.assertFalse(r.ok)

    def test_envelope_only_text_without_deltas_fails(self) -> None:
        # WHY: collapsed non-stream-as-stream must not green the happy path.
        raw = (
            "event: response.completed\n"
            'data: {"type":"response.completed","response":{"status":"completed",'
            '"output_text":"pong"}}\n\n'
        )
        r = O.oracle_responses_stream_happy(
            200, {"Content-Type": "text/event-stream"}, raw
        )
        self.assertFalse(r.ok)
        self.assertIn("delta", r.reason.lower())

    def test_partial_text_then_failed_fails(self) -> None:
        # WHY: partial text + response.failed must not green the happy path.
        raw = (
            "event: response.output_text.delta\n"
            'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
            "event: response.failed\n"
            'data: {"type":"response.failed","error":{"message":"boom"}}\n\n'
        )
        r = O.oracle_responses_stream_happy(
            200, {"Content-Type": "text/event-stream"}, raw
        )
        self.assertFalse(r.ok)
        self.assertIn("failed", r.reason.lower())


class CountTokensOracleTests(unittest.TestCase):
    def test_monotonic(self) -> None:
        r = O.oracle_count_tokens_pair(200, {"input_tokens": 3}, 200, {"input_tokens": 40})
        self.assertTrue(r.ok, r.reason)

    def test_equal_fails(self) -> None:
        r = O.oracle_count_tokens_pair(200, {"input_tokens": 10}, 200, {"input_tokens": 10})
        self.assertFalse(r.ok)

    def test_zero_fails(self) -> None:
        r = O.oracle_count_tokens_pair(200, {"input_tokens": 0}, 200, {"input_tokens": 5})
        self.assertFalse(r.ok)


class DualModeOracleTests(unittest.TestCase):
    def test_anthropic_message_shape(self) -> None:
        body = {
            "type": "message",
            "role": "assistant",
            "model": "grok-4.5",
            "content": [{"type": "text", "text": "pong"}],
        }
        r = O.oracle_anthropic_message(200, body, expected_model_substr="grok-4.5")
        self.assertTrue(r.ok, r.reason)

    def test_empty_content_fails(self) -> None:
        r = O.oracle_anthropic_message(
            200, {"type": "message", "role": "assistant", "content": []}
        )
        self.assertFalse(r.ok)

    def test_missing_type_fails(self) -> None:
        r = O.oracle_anthropic_message(
            200, {"role": "assistant", "content": [{"type": "text", "text": "x"}]}
        )
        self.assertFalse(r.ok)

    def test_empty_text_block_fails(self) -> None:
        r = O.oracle_anthropic_message(
            200,
            {"type": "message", "role": "assistant", "content": [{"type": "text", "text": ""}]},
        )
        self.assertFalse(r.ok)

    def test_claude_model_misroute_fails(self) -> None:
        body = {
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [{"type": "text", "text": "pong"}],
        }
        r = O.oracle_anthropic_message(200, body, expected_model_substr="grok-4.5")
        self.assertFalse(r.ok)

    def test_non_200_fails(self) -> None:
        r = O.oracle_anthropic_message(400, {"type": "error", "error": {"message": "x"}})
        self.assertFalse(r.ok)


class ModelPinTests(unittest.TestCase):
    def test_require_pin_present(self) -> None:
        r = O.require_pin_in_models(
            DEFAULT_CLAUDE_MODEL, {DEFAULT_CLAUDE_MODEL, "grok-4.5"}
        )
        self.assertTrue(r.ok)

    def test_require_pin_missing_fails_loud(self) -> None:
        r = O.require_pin_in_models("claude-haiku-4-5-20251001", {"grok-4.5"})
        self.assertFalse(r.ok)
        self.assertIn("missing", r.reason)

    def test_tool_loop_nonce_and_final(self) -> None:
        self.assertTrue(O.is_digit_nonce("123456"))
        self.assertFalse(O.is_digit_nonce("12345"))  # need >= 6
        self.assertFalse(O.is_digit_nonce("12ab56"))
        body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "The code is 84729103 done."},
                    "finish_reason": "stop",
                }
            ]
        }
        r = O.oracle_final_text_contains_nonce(body, "84729103")
        self.assertTrue(r.ok, r.reason)
        r2 = O.oracle_final_text_contains_nonce(body, "000000")
        self.assertFalse(r2.ok)


class ChatToolCallOracleTests(unittest.TestCase):
    def test_turn1_requires_tool_calls(self) -> None:
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "report_nonce", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        r = O.oracle_chat_tool_calls_nonstream(body, expected_name="report_nonce")
        self.assertTrue(r.ok, r.reason)

    def test_missing_tool_calls_fails(self) -> None:
        body = {
            "choices": [
                {"message": {"role": "assistant", "content": "no tools"}, "finish_reason": "stop"}
            ]
        }
        r = O.oracle_chat_tool_calls_nonstream(body)
        self.assertFalse(r.ok)


class ClientRetryLoopTests(unittest.TestCase):
    """Hermetic coverage of LiveHttpClient.request retry loop (no real network)."""

    def test_retries_503_without_reading_body_then_returns_200(self) -> None:
        from tools.live_http_suite.client import LiveHttpClient

        client = LiveHttpClient("http://127.0.0.1:9", max_transport_retries=3, retry_backoff_s=0)
        calls: list[int] = []
        bodies_read: list[bool] = []

        class FakeReader:
            def __init__(self, body: str) -> None:
                self._body = body
                self.closed = False
                self.bytes_received = 0

            def read_body(self, *, deadline: float) -> str:  # noqa: ARG002
                bodies_read.append(True)
                return self._body

            def close(self) -> None:
                self.closed = True

        def fake_once(method, path, *, data, headers, deadline):  # noqa: ANN001, ARG001
            calls.append(1)
            if len(calls) < 3:
                return 503, {"Content-Type": "application/json"}, FakeReader("retry-me")
            return 200, {"Content-Type": "application/json"}, FakeReader('{"ok":true}')

        client._once_headers = fake_once  # type: ignore[method-assign]
        resp = client.request("GET", "/health")
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.body, '{"ok":true}')
        self.assertEqual(len(calls), 3)
        # First two attempts must abandon body unread (only final read_body).
        self.assertEqual(bodies_read, [True])

    def test_retry_false_does_not_retry_503(self) -> None:
        from tools.live_http_suite.client import LiveHttpClient

        client = LiveHttpClient("http://127.0.0.1:9", max_transport_retries=5, retry_backoff_s=0)
        calls: list[int] = []

        class FakeReader:
            bytes_received = 0

            def read_body(self, *, deadline: float) -> str:  # noqa: ARG002
                return "err"

            def close(self) -> None:
                return None

        def fake_once(method, path, *, data, headers, deadline):  # noqa: ANN001, ARG001
            calls.append(1)
            return 503, {}, FakeReader()

        client._once_headers = fake_once  # type: ignore[method-assign]
        resp = client.request("GET", "/x", retry=False)
        self.assertEqual(resp.status, 503)
        self.assertEqual(len(calls), 1)

    def test_final_503_still_returns_body(self) -> None:
        from tools.live_http_suite.client import LiveHttpClient

        client = LiveHttpClient("http://127.0.0.1:9", max_transport_retries=2, retry_backoff_s=0)

        class FakeReader:
            def __init__(self, body: str) -> None:
                self._body = body
                self.bytes_received = 0

            def read_body(self, *, deadline: float) -> str:  # noqa: ARG002
                return self._body

            def close(self) -> None:
                return None

        n = {"i": 0}

        def fake_once(method, path, *, data, headers, deadline):  # noqa: ANN001, ARG001
            n["i"] += 1
            return 503, {}, FakeReader(f"body-{n['i']}")

        client._once_headers = fake_once  # type: ignore[method-assign]
        resp = client.request("GET", "/x")
        self.assertEqual(resp.status, 503)
        self.assertEqual(resp.body, "body-2")

    def test_no_retry_after_partial_body_bytes(self) -> None:
        # WHY: mid-body failure must not re-issue the request (harness rule).
        from tools.live_http_suite.client import LiveHttpClient, TransportError

        client = LiveHttpClient("http://127.0.0.1:9", max_transport_retries=5, retry_backoff_s=0)
        calls: list[int] = []

        class FakeReader:
            bytes_received = 0

            def read_body(self, *, deadline: float) -> str:  # noqa: ARG002
                self.bytes_received = 12
                raise TransportError("incomplete after 12 bytes")

            def close(self) -> None:
                return None

        def fake_once(method, path, *, data, headers, deadline):  # noqa: ANN001, ARG001
            calls.append(1)
            return 200, {}, FakeReader()

        client._once_headers = fake_once  # type: ignore[method-assign]
        with self.assertRaises(TransportError):
            client.request("GET", "/x")
        self.assertEqual(len(calls), 1, "must not retry after body bytes received")


class SummarizeTests(unittest.TestCase):
    def test_empty_outcomes_exit_nonzero(self) -> None:
        from tools.live_http_suite.suite import TestOutcome, summarize

        code = summarize([])
        self.assertEqual(code, 1)

    def test_all_skip_exit_nonzero(self) -> None:
        from tools.live_http_suite.suite import TestOutcome, summarize

        code = summarize([TestOutcome(name="t", status="skip", detail="off")])
        self.assertEqual(code, 1)

    def test_pass_exit_zero(self) -> None:
        from tools.live_http_suite.suite import TestOutcome, summarize

        code = summarize([TestOutcome(name="t", status="pass", detail="ok")])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
