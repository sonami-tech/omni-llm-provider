"""Pure oracles and SSE helpers for the live HTTP suite.

These functions take already-received HTTP status/headers/body and return
pass/fail reasons. No network I/O. Unit tests exercise them hermetically.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleResult:
    ok: bool
    reason: str

    @classmethod
    def pass_(cls, reason: str = "ok") -> "OracleResult":
        return cls(True, reason)

    @classmethod
    def fail(cls, reason: str) -> "OracleResult":
        return cls(False, reason)


# ---------------------------------------------------------------------------
# Content-type helpers
# ---------------------------------------------------------------------------


def content_type_main(headers: Mapping[str, str]) -> str:
    """Return the media type without parameters (lowercased)."""
    raw = ""
    for key, value in headers.items():
        if key.lower() == "content-type":
            raw = value
            break
    return raw.split(";", 1)[0].strip().lower()


def is_json_content_type(headers: Mapping[str, str]) -> bool:
    ct = content_type_main(headers)
    return ct == "application/json" or ct.endswith("+json")


def is_event_stream_content_type(headers: Mapping[str, str]) -> bool:
    return content_type_main(headers) == "text/event-stream"


def oracle_dropped_tool_choice(headers: Mapping[str, str]) -> OracleResult:
    """Claude sends forced tool choice as `auto` and reports it in `x-omni-dropped`."""
    for key, value in headers.items():
        if key.lower() == "x-omni-dropped":
            tokens = [token.strip() for token in value.split(",")]
            if "tool_choice" in tokens:
                return OracleResult.pass_("x-omni-dropped contains tool_choice")
            return OracleResult.fail(f"x-omni-dropped {value!r} lacks tool_choice")
    return OracleResult.fail("missing x-omni-dropped header")


# ---------------------------------------------------------------------------
# OpenAI error object
# ---------------------------------------------------------------------------


def is_openai_error_object(body: Any) -> bool:
    """True when body is an OpenAI-shaped error envelope."""
    if not isinstance(body, dict):
        return False
    err = body.get("error")
    if not isinstance(err, dict):
        return False
    # Require at least a message string (type is usually present too).
    msg = err.get("message")
    return isinstance(msg, str) and len(msg) > 0


def parse_json_body(raw: str | bytes | None) -> Any | None:
    if raw is None:
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


@dataclass
class SseEvent:
    event: str | None = None
    data: str = ""
    id: str | None = None


def parse_sse(raw: str) -> list[SseEvent]:
    """Parse a complete SSE body into events.

    Blank line separates events. Multi-line data fields are joined with \\n.
    A single leading space after the field colon is stripped (SSE spec).
    """
    events: list[SseEvent] = []
    current = SseEvent()
    data_parts: list[str] = []

    def flush() -> None:
        nonlocal current, data_parts
        if data_parts or current.event is not None or current.id is not None:
            current.data = "\n".join(data_parts)
            events.append(current)
        current = SseEvent()
        data_parts = []

    for line in raw.splitlines():
        if line == "":
            flush()
            continue
        if line.startswith(":"):
            continue  # comment
        if ":" not in line:
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            current.event = value
        elif field == "data":
            data_parts.append(value)
        elif field == "id":
            current.id = value
        # ignore retry: and unknown fields
    flush()
    return events


def sse_data_json_values(events: Sequence[SseEvent]) -> list[Any]:
    """Parse JSON from each event data line that is not [DONE]."""
    out: list[Any] = []
    for ev in events:
        data = ev.data.strip()
        if not data or data == "[DONE]":
            continue
        try:
            out.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return out


def _is_truthy_error_field(value: Any) -> bool:
    """True when a JSON `error` field is a real error payload (not null/false/"")."""
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        # Empty {} is weak; require message or type if present, else any non-empty dict.
        if not value:
            return False
        return True
    return True


def sse_has_top_level_error(events: Sequence[SseEvent]) -> bool:
    """True if any frame is event:error or data JSON has a truthy top-level error."""
    for ev in events:
        if (ev.event or "").lower() == "error":
            return True
        data = ev.data.strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and _is_truthy_error_field(obj.get("error")):
            return True
    return False


# ---------------------------------------------------------------------------
# Chat stream assembly (tool_calls)
# ---------------------------------------------------------------------------


@dataclass
class AssembledToolCall:
    id: str | None = None
    name: str | None = None
    arguments: str = ""
    index: int = 0


@dataclass
class AssembledChatStream:
    tool_calls: dict[int, AssembledToolCall] = field(default_factory=dict)
    content: str = ""
    finish_reasons: list[str] = field(default_factory=list)
    saw_done: bool = False
    had_error_frame: bool = False

    def ordered_tool_calls(self) -> list[AssembledToolCall]:
        return [self.tool_calls[i] for i in sorted(self.tool_calls)]


def assemble_chat_stream(events: Sequence[SseEvent]) -> AssembledChatStream:
    assembled = AssembledChatStream()
    for ev in events:
        data = ev.data.strip()
        if data == "[DONE]":
            assembled.saw_done = True
            continue
        if (ev.event or "").lower() == "error":
            assembled.had_error_frame = True
        if not data:
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if _is_truthy_error_field(obj.get("error")):
            assembled.had_error_frame = True
        choices = obj.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            continue
        fr = choice0.get("finish_reason")
        if isinstance(fr, str) and fr:
            assembled.finish_reasons.append(fr)
        delta = choice0.get("delta") or {}
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            assembled.content += content
        tcs = delta.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            idx = tc.get("index", 0)
            if not isinstance(idx, int):
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    idx = 0
            slot = assembled.tool_calls.setdefault(idx, AssembledToolCall(index=idx))
            if isinstance(tc.get("id"), str) and tc["id"]:
                slot.id = tc["id"]
            fn = tc.get("function") or {}
            if isinstance(fn, dict):
                if isinstance(fn.get("name"), str) and fn["name"]:
                    slot.name = fn["name"]
                args = fn.get("arguments")
                if isinstance(args, str):
                    slot.arguments += args
    return assembled


def assemble_responses_text(events: Sequence[SseEvent]) -> tuple[str, bool, bool, int]:
    """Return (assembled_text, saw_completed, saw_error, text_delta_count).

    Happy-path terminal is response.completed only. response.failed / error set
    saw_error. text_delta_count counts response.output_text.delta frames only
    (envelope text alone is not enough for a happy stream).
    """
    text_parts: list[str] = []
    saw_completed = False
    saw_error = False
    text_delta_count = 0
    for ev in events:
        name = (ev.event or "").strip()
        data = ev.data.strip()
        if name in {"response.failed", "error"}:
            saw_error = True
        if not data or data == "[DONE]":
            # [DONE] alone is not a Responses happy terminal.
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if _is_truthy_error_field(obj.get("error")):
            saw_error = True
        kind = name or (obj.get("type") if isinstance(obj.get("type"), str) else "")
        # Prefer event name for text deltas. Refusals are not happy-path text.
        if kind == "response.output_text.delta":
            delta = obj.get("delta")
            if isinstance(delta, str):
                text_parts.append(delta)
                text_delta_count += 1
            continue
        if kind == "response.refusal.delta":
            continue
        if kind == "response.completed":
            response = obj.get("response")
            if isinstance(response, dict):
                status = response.get("status")
                if isinstance(status, str) and status.lower() in {
                    "failed",
                    "cancelled",
                    "incomplete",
                }:
                    saw_error = True
                else:
                    saw_completed = True
                # Envelope text is diagnostic only; happy path requires deltas.
            else:
                saw_completed = True
        if kind in {"response.failed", "error"}:
            saw_error = True
        # Nested failed status on any event's response object.
        response = obj.get("response")
        if isinstance(response, dict):
            status = response.get("status")
            if isinstance(status, str) and status.lower() == "failed":
                saw_error = True
    return ("".join(text_parts), saw_completed, saw_error, text_delta_count)


def _extract_responses_output_text(response: Mapping[str, Any]) -> str:
    parts: list[str] = []
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        if item.get("type") == "message" and isinstance(item.get("content"), list):
            pass  # already handled
    # Also output_text convenience field if present.
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Test oracles (status + body)
# ---------------------------------------------------------------------------


AUTH_OR_RATE_STATUSES = frozenset({401, 403, 429})


def _error_mentions_unknown_model(body: Any, model: str) -> bool:
    if not isinstance(body, dict):
        return False
    err = body.get("error")
    if not isinstance(err, dict):
        return False
    msg = err.get("message")
    if not isinstance(msg, str):
        return False
    lower = msg.lower()
    # Require the model id or the explicit "unknown model" phrase (not bare
    # "not found", which is too broad for other 404s).
    return model.lower() in lower or "unknown model" in lower


def oracle_unknown_model_nonstream(
    status: int,
    headers: Mapping[str, str],
    body_raw: str,
    *,
    model: str = "no-such-model-xyz",
) -> OracleResult:
    """Test 1: 400/404 + OpenAI error object. Fail auth/rate/5xx/hang (hang is client)."""
    if status in AUTH_OR_RATE_STATUSES:
        return OracleResult.fail(
            f"status {status} is auth/rate-limit, not unknown-model (false green)"
        )
    if status >= 500:
        return OracleResult.fail(f"status {status} is server error, not unknown-model")
    if status not in (400, 404):
        return OracleResult.fail(f"expected 400 or 404, got {status}")
    parsed = parse_json_body(body_raw)
    if not is_openai_error_object(parsed):
        return OracleResult.fail(
            f"body is not an OpenAI-shaped error object "
            f"(content-type={content_type_main(headers)!r})"
        )
    if not _error_mentions_unknown_model(parsed, model):
        return OracleResult.fail(
            "error message does not mention unknown model / model id "
            "(rejecting generic 400 as false green)"
        )
    return OracleResult.pass_("400/404 + OpenAI unknown-model error object")


def oracle_stream_x_error(
    status: int,
    headers: Mapping[str, str],
    body_raw: str,
    *,
    model: str = "no-such-model-xyz",
) -> OracleResult:
    """Test 2: (a) 400/404 error JSON or (b) 200 event-stream with error frame.

    Pass requires an error signal AND no normal successful completion content.
    A mixed stream (error frame plus stop+content) fails.
    """
    if status in AUTH_OR_RATE_STATUSES:
        return OracleResult.fail(
            f"status {status} is auth/rate-limit, not stream×error (false green)"
        )
    if status >= 500:
        return OracleResult.fail(f"status {status} is server error")

    # Path (a): early JSON error
    if status in (400, 404):
        parsed = parse_json_body(body_raw)
        if not is_openai_error_object(parsed):
            return OracleResult.fail("400/404 without OpenAI error object")
        if not _error_mentions_unknown_model(parsed, model):
            return OracleResult.fail(
                "error message does not mention unknown model / model id"
            )
        return OracleResult.pass_("400/404 + error JSON (no stream body)")

    # Path (b): 200 + event-stream + error frame + no normal success
    if status != 200:
        return OracleResult.fail(f"expected 400/404 or 200, got {status}")
    if not is_event_stream_content_type(headers):
        return OracleResult.fail(
            f"200 without text/event-stream (got {content_type_main(headers)!r})"
        )
    if not body_raw.strip():
        return OracleResult.fail("empty stream body (silent empty close)")

    events = parse_sse(body_raw)
    if not events:
        return OracleResult.fail("no SSE events parsed from stream body")

    assembled = assemble_chat_stream(events)
    has_error = (
        sse_has_top_level_error(events)
        or assembled.had_error_frame
        or "error" in assembled.finish_reasons
    )
    if not has_error:
        return OracleResult.fail("stream closed without any error frame")

    # Fail normal success terminals even when an error key also appeared.
    if any(fr in {"stop", "length"} for fr in assembled.finish_reasons):
        return OracleResult.fail(
            "stream has success finish_reason stop/length (not a pure error path)"
        )
    # Tool success (even with a parallel error frame) is not a pure error path.
    if "tool_calls" in assembled.finish_reasons and assembled.ordered_tool_calls():
        return OracleResult.fail("stream completed with tool_calls success, not pure error")

    return OracleResult.pass_("200 event-stream with error frame and clean close")


def oracle_chat_tool_calls_nonstream(body: Any, *, expected_name: str | None = None) -> OracleResult:
    """Turn-1 non-stream must include tool_calls with id + name."""
    if not isinstance(body, dict):
        return OracleResult.fail("response is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return OracleResult.fail("missing choices")
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return OracleResult.fail("missing message")
    tcs = msg.get("tool_calls")
    if not isinstance(tcs, list) or not tcs:
        return OracleResult.fail("turn1 missing tool_calls")
    primary = tcs[0]
    if not isinstance(primary, dict):
        return OracleResult.fail("tool_calls[0] is not an object")
    tid = primary.get("id")
    if not isinstance(tid, str) or not tid.strip():
        return OracleResult.fail("tool_calls[0] missing non-empty id")
    fn = primary.get("function") if isinstance(primary.get("function"), dict) else None
    name = fn.get("name") if isinstance(fn, dict) else None
    if not isinstance(name, str) or not name.strip():
        return OracleResult.fail("tool_calls[0] missing function.name")
    if expected_name is not None and name != expected_name:
        return OracleResult.fail(f"tool name {name!r} != {expected_name!r}")
    return OracleResult.pass_("tool_calls present with id and name")


def oracle_final_text_contains_nonce(body: Any, nonce: str) -> OracleResult:
    if not isinstance(body, dict):
        return OracleResult.fail("response is not a JSON object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return OracleResult.fail("missing choices")
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return OracleResult.fail("missing message")
    content = msg.get("content")
    if not isinstance(content, str) or not content:
        return OracleResult.fail("empty final content")
    if nonce not in content:
        return OracleResult.fail(f"final text does not contain nonce {nonce}")
    return OracleResult.pass_("final text contains nonce")


def oracle_stream_forced_tool(
    status: int,
    headers: Mapping[str, str],
    body_raw: str,
    *,
    expected_name: str,
) -> OracleResult:
    """Test 4: assembled tool_calls name match, args JSON object, terminal tool finish."""
    if status != 200:
        return OracleResult.fail(f"expected 200, got {status}")
    if not is_event_stream_content_type(headers):
        return OracleResult.fail(
            f"expected text/event-stream, got {content_type_main(headers)!r}"
        )
    events = parse_sse(body_raw)
    assembled = assemble_chat_stream(events)
    if assembled.had_error_frame or "error" in assembled.finish_reasons:
        return OracleResult.fail("stream contains error frame (not a clean forced-tool path)")
    if not assembled.saw_done:
        return OracleResult.fail("chat stream missing data: [DONE] terminal sentinel")
    tcs = assembled.ordered_tool_calls()
    if not tcs:
        return OracleResult.fail("no tool_calls assembled from stream")
    primary = tcs[0]
    if primary.name != expected_name:
        return OracleResult.fail(f"tool name {primary.name!r} != {expected_name!r}")
    if not primary.arguments:
        return OracleResult.fail("empty tool arguments")
    try:
        args = json.loads(primary.arguments)
    except json.JSONDecodeError as e:
        return OracleResult.fail(f"tool arguments not valid JSON: {e}")
    if not isinstance(args, dict):
        return OracleResult.fail(
            f"tool arguments JSON is not an object: {type(args).__name__}"
        )
    terminal = assembled.finish_reasons
    if not terminal:
        return OracleResult.fail("missing terminal finish_reason")
    if "tool_calls" not in terminal:
        return OracleResult.fail(
            f"expected finish_reason tool_calls among {terminal}, got no tool finish"
        )
    return OracleResult.pass_("stream forced tool: name, args JSON, terminal tool finish")


def oracle_responses_stream_happy(
    status: int,
    headers: Mapping[str, str],
    body_raw: str,
) -> OracleResult:
    """Test 5: non-empty assembled text + response.completed + event-stream CT.

    Any error/failed frame fails even if partial text was emitted.
    """
    if status != 200:
        return OracleResult.fail(f"expected 200, got {status}")
    if not is_event_stream_content_type(headers):
        return OracleResult.fail(
            f"expected text/event-stream, got {content_type_main(headers)!r}"
        )
    events = parse_sse(body_raw)
    text, saw_completed, saw_error, delta_count = assemble_responses_text(events)
    if saw_error:
        return OracleResult.fail("responses stream contains failed/error event")
    if delta_count < 1:
        return OracleResult.fail(
            "no response.output_text.delta frames (envelope-only is not a happy stream)"
        )
    if not text.strip():
        return OracleResult.fail("metadata-only stream with empty assembled text")
    if not saw_completed:
        return OracleResult.fail("stream closed without response.completed")
    return OracleResult.pass_("responses stream: non-empty text deltas + completed")


def oracle_count_tokens_pair(
    short_status: int,
    short_body: Any,
    long_status: int,
    long_body: Any,
) -> OracleResult:
    """Test 6: both 200, input_tokens > 0, long > short."""
    if short_status != 200:
        return OracleResult.fail(f"short count_tokens status {short_status}")
    if long_status != 200:
        return OracleResult.fail(f"long count_tokens status {long_status}")
    if not isinstance(short_body, dict) or not isinstance(long_body, dict):
        return OracleResult.fail("count_tokens body is not a JSON object")
    def _int_tokens(body: dict, label: str) -> int | OracleResult:
        raw = body.get("input_tokens")
        if type(raw) is not int:  # reject bool/float/str via exact type
            return OracleResult.fail(f"{label} input_tokens must be a JSON integer, got {raw!r}")
        return raw

    short_n = _int_tokens(short_body, "short")
    if isinstance(short_n, OracleResult):
        return short_n
    long_n = _int_tokens(long_body, "long")
    if isinstance(long_n, OracleResult):
        return long_n
    if short_n <= 0:
        return OracleResult.fail(f"short input_tokens must be > 0, got {short_n}")
    if long_n <= 0:
        return OracleResult.fail(f"long input_tokens must be > 0, got {long_n}")
    if not (long_n > short_n):
        return OracleResult.fail(f"expected long > short, got {long_n} <= {short_n}")
    return OracleResult.pass_(f"monotonic input_tokens short={short_n} long={long_n}")


def oracle_anthropic_message(
    status: int,
    body: Any,
    *,
    expected_model_substr: str | None = None,
) -> OracleResult:
    """Test 7: 200 Anthropic message shape + non-empty content.

    optional expected_model_substr fails when the response model clearly belongs
    to another family (misroute detection for dual-mode).
    """
    if status != 200:
        return OracleResult.fail(f"expected 200, got {status}")
    if not isinstance(body, dict):
        return OracleResult.fail("body is not a JSON object")
    # Anthropic message requires type=message and role=assistant.
    msg_type = body.get("type")
    if msg_type != "message":
        return OracleResult.fail(f"expected type=message, got {msg_type!r}")
    role = body.get("role")
    if role != "assistant":
        return OracleResult.fail(f"expected role=assistant, got {role!r}")
    if expected_model_substr:
        model = body.get("model")
        if isinstance(model, str) and model:
            # Fail loud on clear Claude ids when dual-mode pin is Grok-family.
            lower = model.lower()
            needle = expected_model_substr.lower()
            if needle.startswith("grok") and "claude" in lower:
                return OracleResult.fail(
                    f"dual-mode expected grok-family model, response model={model!r}"
                )
            if needle.startswith("claude") and "grok" in lower:
                return OracleResult.fail(
                    f"dual-mode expected claude-family model, response model={model!r}"
                )
    content = body.get("content")
    if content is None:
        return OracleResult.fail("missing content")
    if isinstance(content, str):
        if not content.strip():
            return OracleResult.fail("empty string content")
        return OracleResult.pass_("Anthropic message with string content")
    if isinstance(content, list):
        if not content:
            return OracleResult.fail("empty content array")
        # Non-empty means real payload: non-blank text, or a tool_use with name.
        nonempty = False
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" or "text" in block:
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    nonempty = True
                    break
                # Explicit empty text block is not content.
                if btype == "text":
                    continue
            if btype == "tool_use":
                name = block.get("name")
                if isinstance(name, str) and name.strip():
                    nonempty = True
                    break
            if btype == "thinking":
                thinking = block.get("thinking")
                if isinstance(thinking, str) and thinking.strip():
                    nonempty = True
                    break
        if not nonempty:
            return OracleResult.fail("content blocks are empty")
        return OracleResult.pass_("Anthropic message with non-empty content")
    return OracleResult.fail(f"unexpected content type {type(content).__name__}")


def model_ids_from_models_response(body: Any) -> set[str]:
    ids: set[str] = set()
    if not isinstance(body, dict):
        return ids
    data = body.get("data")
    if not isinstance(data, list):
        return ids
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            ids.add(item["id"])
    return ids


def require_pin_in_models(pin: str, model_ids: Iterable[str]) -> OracleResult:
    """Fail loud if required pin is missing from /v1/models.

    Accept exact id match. Prefix/catalog may list dated ids; exact is required
    for the pin string itself (issue: fail loud if required pin missing).
    """
    ids = set(model_ids)
    if pin in ids:
        return OracleResult.pass_(f"pin {pin} present in /v1/models")
    # Also accept if any listed id equals after ignoring provider prefix.
    bare = pin.split(":", 1)[-1]
    if bare in ids:
        return OracleResult.pass_(f"pin {bare} present in /v1/models")
    return OracleResult.fail(
        f"required model pin {pin!r} missing from /v1/models ({len(ids)} models listed)"
    )


_DIGIT_NONCE_RE = re.compile(r"^\d{6,}$")


def is_digit_nonce(value: str) -> bool:
    return bool(_DIGIT_NONCE_RE.match(value))
