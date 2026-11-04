"""Seven live HTTP tests against a running omni instance (issue #15)."""

from __future__ import annotations

import secrets
import sys
import time
from dataclasses import dataclass
from typing import Callable

from tools.live_http_suite.client import HttpResponse, LiveHttpClient, TransportError
from tools.live_http_suite.config import UNKNOWN_MODEL, SuiteConfig
from tools.live_http_suite import oracles as O


@dataclass
class TestOutcome:
    name: str
    status: str  # "pass" | "fail" | "skip"
    detail: str
    elapsed_s: float = 0.0


class SuiteFailure(Exception):
    """A single test failed its oracle or transport."""


def _nonce_digits(n: int = 8) -> str:
    # Digit-only nonce, at least 6 digits. Avoid leading zeros so a model that
    # strips them still matches the literal nonce string in the final text.
    if n < 6:
        n = 6
    low = 10 ** (n - 1)
    high = 10**n
    value = secrets.randbelow(high - low) + low
    out = str(value)
    assert O.is_digit_nonce(out)
    return out


class LiveHttpSuite:
    def __init__(self, cfg: SuiteConfig, client: LiveHttpClient | None = None) -> None:
        self.cfg = cfg
        self.client = client or LiveHttpClient(
            cfg.base_url_normalized,
            timeout_s=cfg.timeout_s,
            connect_timeout_s=cfg.connect_timeout_s,
            max_transport_retries=cfg.max_transport_retries,
        )
        self._model_ids: set[str] | None = None

    # -- helpers -------------------------------------------------------------

    def fetch_models(self) -> set[str]:
        resp = self.client.get_json("/v1/models")
        if resp.status != 200:
            raise SuiteFailure(f"GET /v1/models returned {resp.status}: {resp.body[:300]}")
        body = O.parse_json_body(resp.body)
        ids = O.model_ids_from_models_response(body)
        self._model_ids = ids
        return ids

    def require_pin(self, pin: str) -> None:
        ids = self._model_ids if self._model_ids is not None else self.fetch_models()
        result = O.require_pin_in_models(pin, ids)
        if not result.ok:
            raise SuiteFailure(result.reason)

    def _post(self, path: str, body: dict, *, retry: bool = True) -> HttpResponse:
        try:
            return self.client.post_json(path, body, retry=retry)
        except TransportError as e:
            raise SuiteFailure(f"transport error on POST {path}: {e}") from e

    # -- tests ---------------------------------------------------------------

    def test_unknown_model_nonstream(self) -> None:
        """1. Unknown model non-stream → 400/404 + OpenAI error object."""
        body = {
            "model": UNKNOWN_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        }
        resp = self._post("/v1/chat/completions", body)
        result = O.oracle_unknown_model_nonstream(
            resp.status, resp.headers, resp.body, model=UNKNOWN_MODEL
        )
        if not result.ok:
            raise SuiteFailure(result.reason)

    def test_stream_x_error(self) -> None:
        """2. Stream × error for unknown model."""
        body = {
            "model": UNKNOWN_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": True,
        }
        resp = self._post("/v1/chat/completions", body)
        result = O.oracle_stream_x_error(
            resp.status, resp.headers, resp.body, model=UNKNOWN_MODEL
        )
        if not result.ok:
            raise SuiteFailure(result.reason)

    def test_claude_tool_loop_nonstream(self) -> None:
        """3. Claude tool loop non-stream with digit nonce.

        Claude receives the named choice as `auto` (issue #63), so the tool call
        relies on the prompt wording and the reply must report `tool_choice`.
        """
        self.require_pin(self.cfg.claude_model)
        nonce = _nonce_digits(8)
        tool_name = "report_nonce"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Report a diagnostic nonce to the user path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "note": {"type": "string", "description": "optional note"},
                        },
                    },
                },
            }
        ]
        # Turn 1: name the tool; Omni sends it to Claude as `auto`.
        body1 = {
            "model": self.cfg.claude_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call the report_nonce tool now. Do not answer in plain text first."
                    ),
                }
            ],
            "tools": tools,
            "tool_choice": {
                "type": "function",
                "function": {"name": tool_name},
            },
            "max_tokens": 256,
        }
        resp1 = self._post("/v1/chat/completions", body1, retry=False)
        if resp1.status != 200:
            raise SuiteFailure(f"turn1 status {resp1.status}: {resp1.body[:400]}")
        dropped1 = O.oracle_dropped_tool_choice(resp1.headers)
        if not dropped1.ok:
            raise SuiteFailure(f"turn1: {dropped1.reason}")
        parsed1 = O.parse_json_body(resp1.body)
        r1 = O.oracle_chat_tool_calls_nonstream(parsed1, expected_name=tool_name)
        if not r1.ok:
            raise SuiteFailure(f"turn1: {r1.reason}; body={resp1.body[:500]}")

        assert isinstance(parsed1, dict)
        choice0 = parsed1["choices"][0]
        assistant_msg = choice0["message"]
        tool_calls = assistant_msg["tool_calls"]
        tool_call_id = tool_calls[0].get("id") or "call_nonce_1"

        # Turn 2: tool result carries the digit nonce AND the instruction to
        # echo it. Do NOT append a trailing user message after role:tool —
        # Claude translate maps tool → user and rejects adjacent user turns
        # that are not coalesced (would be a live 400).
        body2 = {
            "model": self.cfg.claude_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call the report_nonce tool now. Do not answer in plain text first. "
                        "After the tool returns, reply with ONLY the digit nonce and nothing else."
                    ),
                },
                {
                    "role": "assistant",
                    "content": assistant_msg.get("content"),
                    "tool_calls": tool_calls,
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": (
                        f"nonce={nonce}. Reply with ONLY that digit nonce and nothing else."
                    ),
                },
            ],
            "tools": tools,
            "tool_choice": "none",
            "max_tokens": 64,
        }
        resp2 = self._post("/v1/chat/completions", body2, retry=False)
        if resp2.status != 200:
            raise SuiteFailure(f"turn2 status {resp2.status}: {resp2.body[:400]}")
        parsed2 = O.parse_json_body(resp2.body)
        r2 = O.oracle_final_text_contains_nonce(parsed2, nonce)
        if not r2.ok:
            raise SuiteFailure(f"turn2: {r2.reason}; body={resp2.body[:500]}")

    def test_stream_forced_tool(self) -> None:
        """4. Stream + named tool on haiku.

        Claude receives the named choice as `auto` (issue #63), so the tool call
        relies on the prompt wording and the reply must report `tool_choice`.
        """
        self.require_pin(self.cfg.claude_model)
        tool_name = "get_weather"
        body = {
            "model": self.cfg.claude_model,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Paris? Use the get_weather tool.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": tool_name},
            },
            "stream": True,
            "max_tokens": 256,
        }
        resp = self._post("/v1/chat/completions", body, retry=False)
        dropped = O.oracle_dropped_tool_choice(resp.headers)
        if resp.status == 200 and not dropped.ok:
            raise SuiteFailure(dropped.reason)
        result = O.oracle_stream_forced_tool(
            resp.status, resp.headers, resp.body, expected_name=tool_name
        )
        if not result.ok:
            raise SuiteFailure(f"{result.reason}; body[:500]={resp.body[:500]!r}")

    def test_responses_stream_happy(self) -> None:
        """5. Responses stream happy path."""
        self.require_pin(self.cfg.responses_model)
        body = {
            "model": self.cfg.responses_model,
            "input": "Reply with exactly the word pong and nothing else.",
            "stream": True,
            "max_output_tokens": 64,
        }
        resp = self._post("/v1/responses", body, retry=False)
        result = O.oracle_responses_stream_happy(resp.status, resp.headers, resp.body)
        if not result.ok:
            raise SuiteFailure(f"{result.reason}; body[:500]={resp.body[:500]!r}")

    def test_count_tokens_monotonic(self) -> None:
        """6. count_tokens short vs long: both 200, long input_tokens > short."""
        self.require_pin(self.cfg.claude_model)
        short_body = {
            "model": self.cfg.claude_model,
            "messages": [{"role": "user", "content": "hi"}],
        }
        long_prompt = "Please summarize the following text. " + ("lorem ipsum " * 80)
        long_body = {
            "model": self.cfg.claude_model,
            "messages": [{"role": "user", "content": long_prompt}],
        }
        # Anthropic count_tokens often wants anthropic-version header; omni may
        # accept without it when --no-auth. Send a standard version for fidelity.
        headers = {"anthropic-version": "2023-06-01"}
        try:
            s_resp = self.client.post_json(
                "/v1/messages/count_tokens", short_body, headers=headers, retry=False
            )
            l_resp = self.client.post_json(
                "/v1/messages/count_tokens", long_body, headers=headers, retry=False
            )
        except TransportError as e:
            raise SuiteFailure(f"transport error on count_tokens: {e}") from e
        result = O.oracle_count_tokens_pair(
            s_resp.status,
            O.parse_json_body(s_resp.body),
            l_resp.status,
            O.parse_json_body(l_resp.body),
        )
        if not result.ok:
            raise SuiteFailure(
                f"{result.reason}; short={s_resp.status}:{s_resp.body[:200]!r} "
                f"long={l_resp.status}:{l_resp.body[:200]!r}"
            )

    def test_dual_mode_anthropic_edge(self) -> None:
        """7. Dual-mode Anthropic edge with OpenAI-family model (Grok).

        Skip ONLY when the runner set dual-mode-off before the request.
        Never infer skip from 4xx.
        """
        if self.cfg.dual_mode_off:
            raise _Skip(
                "dual-mode-off flag set before request (OMNI_TEST_DUAL_MODE_OFF / --dual-mode-off)"
            )
        # Resolve pin only AFTER the skip flag check.
        self.require_pin(self.cfg.dual_mode_model)
        body = {
            "model": self.cfg.dual_mode_model,
            "max_tokens": 64,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly the word pong and nothing else.",
                }
            ],
        }
        headers = {"anthropic-version": "2023-06-01"}
        try:
            resp = self.client.post_json(
                "/v1/messages", body, headers=headers, retry=False
            )
        except TransportError as e:
            raise SuiteFailure(f"transport error on /v1/messages: {e}") from e
        result = O.oracle_anthropic_message(
            resp.status,
            O.parse_json_body(resp.body),
            expected_model_substr=self.cfg.dual_mode_model,
        )
        if not result.ok:
            raise SuiteFailure(f"{result.reason}; status={resp.status} body={resp.body[:500]!r}")


class _Skip(Exception):
    """Internal skip signal (not a failure)."""


TEST_ORDER: list[tuple[str, Callable[[LiveHttpSuite], None]]] = [
    ("unknown_model_nonstream", LiveHttpSuite.test_unknown_model_nonstream),
    ("stream_x_error", LiveHttpSuite.test_stream_x_error),
    ("claude_tool_loop_nonstream", LiveHttpSuite.test_claude_tool_loop_nonstream),
    ("stream_forced_tool", LiveHttpSuite.test_stream_forced_tool),
    ("responses_stream_happy", LiveHttpSuite.test_responses_stream_happy),
    ("count_tokens_monotonic", LiveHttpSuite.test_count_tokens_monotonic),
    ("dual_mode_anthropic_edge", LiveHttpSuite.test_dual_mode_anthropic_edge),
]


def known_test_names() -> set[str]:
    return {name for name, _ in TEST_ORDER}


def run_suite(cfg: SuiteConfig, *, only: set[str] | None = None) -> list[TestOutcome]:
    suite = LiveHttpSuite(cfg)
    outcomes: list[TestOutcome] = []
    if only is not None:
        unknown = sorted(only - known_test_names())
        if unknown:
            return [
                TestOutcome(
                    name="__only__",
                    status="fail",
                    detail=f"unknown test name(s): {', '.join(unknown)}",
                )
            ]
        if not only:
            return [
                TestOutcome(
                    name="__only__",
                    status="fail",
                    detail="--only selected zero tests",
                )
            ]
    # Probe /health first so a dead base URL fails loud.
    try:
        health = suite.client.get_json("/health")
        if health.status != 200:
            print(
                f"WARN: GET /health returned {health.status}; continuing anyway",
                file=sys.stderr,
            )
    except TransportError as e:
        print(f"ERROR: cannot reach {cfg.base_url_normalized}: {e}", file=sys.stderr)
        return [
            TestOutcome(
                name="__connect__",
                status="fail",
                detail=str(e),
            )
        ]

    try:
        suite.fetch_models()
    except (SuiteFailure, TransportError) as e:
        print(f"ERROR: GET /v1/models failed: {e}", file=sys.stderr)
        return [TestOutcome(name="__models__", status="fail", detail=str(e))]

    for name, method in TEST_ORDER:
        if only is not None and name not in only:
            continue
        t0 = time.monotonic()
        try:
            method(suite)
            elapsed = time.monotonic() - t0
            outcomes.append(TestOutcome(name=name, status="pass", detail="ok", elapsed_s=elapsed))
            print(f"PASS  {name}  ({elapsed:.2f}s)")
        except _Skip as e:
            elapsed = time.monotonic() - t0
            outcomes.append(
                TestOutcome(name=name, status="skip", detail=str(e), elapsed_s=elapsed)
            )
            print(f"SKIP  {name}  ({e})")
        except SuiteFailure as e:
            elapsed = time.monotonic() - t0
            outcomes.append(
                TestOutcome(name=name, status="fail", detail=str(e), elapsed_s=elapsed)
            )
            print(f"FAIL  {name}  ({elapsed:.2f}s): {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — surface unexpected errors as fail
            elapsed = time.monotonic() - t0
            outcomes.append(
                TestOutcome(
                    name=name,
                    status="fail",
                    detail=f"unexpected: {type(e).__name__}: {e}",
                    elapsed_s=elapsed,
                )
            )
            print(f"FAIL  {name}  unexpected: {e}", file=sys.stderr)
    return outcomes


def summarize(outcomes: list[TestOutcome]) -> int:
    """Print summary. Return process exit code (0 only if ≥1 pass and 0 fail)."""
    n_pass = sum(1 for o in outcomes if o.status == "pass")
    n_fail = sum(1 for o in outcomes if o.status == "fail")
    n_skip = sum(1 for o in outcomes if o.status == "skip")
    print(f"\nSummary: {n_pass} pass, {n_fail} fail, {n_skip} skip")
    if n_fail:
        return 1
    if n_pass == 0:
        # Zero executed passes (all skip or empty) is not a green suite run.
        print("ERROR: no tests passed (refuse empty/all-skip success)", file=sys.stderr)
        return 1
    return 0
