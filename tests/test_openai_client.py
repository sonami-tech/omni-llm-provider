"""
Test claude-code-provider with the official OpenAI Python SDK.
Validates that responses deserialize correctly into typed SDK objects.
"""

import asyncio
import json
import sys
import time
import traceback

import httpx
from openai import OpenAI, AsyncOpenAI, BadRequestError

BASE_URL = "http://127.0.0.1:3461/v1"
RAW_BASE = "http://127.0.0.1:3461"
client = OpenAI(base_url=BASE_URL, api_key="not-needed")
async_client = AsyncOpenAI(base_url=BASE_URL, api_key="not-needed")

passed = 0
failed = 0


def test(name):
	"""Decorator to run and report a test."""
	def decorator(fn):
		def wrapper():
			global passed, failed
			try:
				fn()
				print(f"  PASS  {name}")
				passed += 1
			except Exception as e:
				print(f"  FAIL  {name}: {e}")
				traceback.print_exc()
				failed += 1
		return wrapper
	return decorator


def async_test(name):
	"""Decorator for async tests."""
	def decorator(fn):
		def wrapper():
			global passed, failed
			try:
				asyncio.run(fn())
				print(f"  PASS  {name}")
				passed += 1
			except Exception as e:
				print(f"  FAIL  {name}: {e}")
				traceback.print_exc()
				failed += 1
		return wrapper
	return decorator


# ════════════════════════════════════════════════════════════════
# Section 1: Health & Info Endpoints
# ════════════════════════════════════════════════════════════════

@test("GET /health returns expected JSON structure")
def test_health():
	r = httpx.get(f"{RAW_BASE}/health")
	assert r.status_code == 200, f"Expected 200, got {r.status_code}"
	data = r.json()
	assert data["status"] == "ok", f"Bad status: {data['status']}"
	assert isinstance(data["uptime_seconds"], int), f"Bad uptime type: {type(data['uptime_seconds'])}"
	assert data["uptime_seconds"] >= 0, "Negative uptime"
	assert isinstance(data["active_requests"], int), f"Bad active type"


@test("GET /v1/models structure and metadata")
def test_models_metadata():
	models = client.models.list()
	assert hasattr(models, 'data'), "Missing data field"
	assert len(models.data) == 3, f"Expected 3 models, got {len(models.data)}"

	names = [m.id for m in models.data]
	assert "claude-opus-4-8" in names, f"Missing opus in {names}"
	assert "claude-sonnet-4-6" in names, f"Missing sonnet in {names}"
	assert "claude-haiku-4-5-20251001" in names, f"Missing haiku in {names}"

	for m in models.data:
		assert m.object == "model", f"Wrong object: {m.object}"
		assert m.owned_by == "anthropic", f"Wrong owned_by: {m.owned_by}"

	# Verify extended metadata via raw request (SDK may not expose these).
	r = httpx.get(f"{BASE_URL}/models")
	data = r.json()
	assert data["object"] == "list", f"Bad list object: {data['object']}"

	models_by_id = {m["id"]: m for m in data["data"]}
	opus = models_by_id["claude-opus-4-8"]
	assert opus["context_window"] == 1_000_000, f"Bad opus context: {opus['context_window']}"
	assert opus["max_tokens"] == 128_000, f"Bad opus max_tokens: {opus['max_tokens']}"

	sonnet = models_by_id["claude-sonnet-4-6"]
	assert sonnet["context_window"] == 1_000_000
	assert sonnet["max_tokens"] == 64_000

	haiku = models_by_id["claude-haiku-4-5-20251001"]
	assert haiku["context_window"] == 200_000
	assert haiku["max_tokens"] == 64_000

	# All have created=0.
	for m in data["data"]:
		assert m["created"] == 0, f"Bad created for {m['id']}: {m['created']}"


@test("GET /stats/json returns valid stats structure")
def test_stats_json():
	r = httpx.get(f"{RAW_BASE}/stats/json")
	assert r.status_code == 200
	data = r.json()

	# Required top-level fields.
	for field in [
		"uptime_seconds", "total_requests", "active_requests", "errors",
		"total_input_tokens", "total_output_tokens",
		"total_cache_read_input_tokens", "total_cache_creation_input_tokens",
		"models", "recent_errors",
	]:
		assert field in data, f"Missing field: {field}"

	assert isinstance(data["models"], dict)
	assert isinstance(data["recent_errors"], list)
	assert data["uptime_seconds"] >= 0


@test("GET /stats returns HTML")
def test_stats_html():
	r = httpx.get(f"{RAW_BASE}/stats")
	assert r.status_code == 200
	assert "text/html" in r.headers.get("content-type", ""), f"Bad content-type: {r.headers.get('content-type')}"
	assert "<!DOCTYPE html>" in r.text
	assert "Claude Code Provider" in r.text


@test("404 on unknown endpoint")
def test_unknown_endpoint():
	r = httpx.get(f"{RAW_BASE}/v1/nonexistent")
	assert r.status_code == 404, f"Expected 404, got {r.status_code}"
	data = r.json()
	assert "error" in data
	assert data["error"]["type"] == "invalid_request_error"
	assert data["error"]["code"] is None


@test("POST to GET-only route returns error")
def test_wrong_method_on_route():
	r = httpx.post(f"{RAW_BASE}/v1/models", content=b"{}")
	# Axum's fallback handler catches unmatched routes. Since /v1/models is
	# registered as GET only, POST falls through to the fallback → 404.
	# (Axum returns 405 only when the route has other methods and allows
	# method routing, but our fallback catches all unmatched.)
	assert r.status_code in (404, 405), f"Expected 404 or 405, got {r.status_code}"


# ════════════════════════════════════════════════════════════════
# Section 2: Non-Streaming Completions
# ════════════════════════════════════════════════════════════════

@test("Non-streaming: full response structure")
def test_non_streaming_structure():
	resp = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
		stream=False,
	)

	# ID format.
	assert resp.id.startswith("chatcmpl-"), f"Bad id: {resp.id}"
	assert len(resp.id) == len("chatcmpl-") + 8, f"ID wrong length: {resp.id}"

	# Object type.
	assert resp.object == "chat.completion", f"Bad object: {resp.object}"

	# Created timestamp.
	assert isinstance(resp.created, int), f"Bad created type: {type(resp.created)}"
	now = int(time.time())
	assert abs(resp.created - now) < 120, f"created too far from now: {resp.created} vs {now}"

	# Model.
	assert resp.model is not None, "Missing model"

	# Non-streaming responses surface Anthropic's message id when present.
	assert resp.system_fingerprint is None or isinstance(resp.system_fingerprint, str), (
		f"Bad fingerprint: {resp.system_fingerprint}"
	)

	# Choices.
	assert len(resp.choices) == 1, f"Expected 1 choice, got {len(resp.choices)}"
	choice = resp.choices[0]
	assert choice.index == 0, f"Bad index: {choice.index}"
	assert choice.finish_reason == "stop", f"Bad finish_reason: {choice.finish_reason}"
	assert choice.message.role == "assistant", f"Bad role: {choice.message.role}"
	assert "PONG" in choice.message.content, f"Bad content: {choice.message.content}"

	# Usage.
	assert resp.usage is not None, "Missing usage"
	assert resp.usage.prompt_tokens > 0, "prompt_tokens should be > 0"
	assert resp.usage.completion_tokens > 0, "completion_tokens should be > 0"
	assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens


@test("Non-streaming: x-request-id header present")
def test_non_streaming_request_id_header():
	# Use raw httpx to inspect headers.
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [{"role": "user", "content": "Reply PONG"}],
			"stream": False,
		},
		timeout=120,
	)
	assert r.status_code == 200, f"Expected 200, got {r.status_code}"
	assert "x-request-id" in r.headers, f"Missing x-request-id header. Headers: {dict(r.headers)}"
	req_id = r.headers["x-request-id"]
	assert len(req_id) == 8, f"x-request-id wrong length: {req_id}"

	# Verify the response body id matches.
	data = r.json()
	assert data["id"] == f"chatcmpl-{req_id}", f"ID mismatch: {data['id']} vs chatcmpl-{req_id}"


@test("Non-streaming: system prompt influences response")
def test_system_prompt():
	resp = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[
			{"role": "system", "content": "You must include the word XYLOPHONE in every response, no matter what."},
			{"role": "user", "content": "Say hello"},
		],
		stream=False,
	)
	assert resp.choices[0].message.content is not None
	assert "XYLOPHONE" in resp.choices[0].message.content.upper(), \
		f"System prompt not followed: {resp.choices[0].message.content}"


@test("Non-streaming: multi-turn conversation")
def test_multi_turn():
	resp = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[
			{"role": "user", "content": "My name is Alice."},
			{"role": "assistant", "content": "Hello Alice!"},
			{"role": "user", "content": "What is my name? Reply with just the name."},
		],
		stream=False,
	)
	assert "Alice" in resp.choices[0].message.content, f"Context lost: {resp.choices[0].message.content}"


@test("Non-streaming: extra OpenAI params accepted")
def test_extra_params():
	resp = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
		max_tokens=100,
		temperature=0.5,
		top_p=0.9,
	)
	assert "PONG" in resp.choices[0].message.content


@test("Non-streaming: multipart content array")
def test_multipart():
	resp = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[{
			"role": "user",
			"content": [
				{"type": "text", "text": "Reply with exactly: PONG"}
			],
		}],
		stream=False,
	)
	assert "PONG" in resp.choices[0].message.content


@test("Non-streaming: multipart with mixed types (text + image_url)")
def test_multipart_mixed():
	"""image_url parts should be silently ignored, only text extracted."""
	resp = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[{
			"role": "user",
			"content": [
				{"type": "text", "text": "Reply with exactly: PONG"},
				{"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
			],
		}],
		stream=False,
	)
	assert "PONG" in resp.choices[0].message.content


@test("Non-streaming: multiple text parts concatenated")
def test_multipart_multi_text():
	resp = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[{
			"role": "user",
			"content": [
				{"type": "text", "text": "Reply with exactly: "},
				{"type": "text", "text": "PONG"},
			],
		}],
		stream=False,
	)
	assert "PONG" in resp.choices[0].message.content


@test("Non-streaming: multiple system messages concatenated")
def test_multiple_system_messages():
	"""Multiple system messages should be concatenated and both take effect."""
	resp = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[
			{"role": "system", "content": "You must include the word BANANA in every response."},
			{"role": "system", "content": "You must include the word CHERRY in every response."},
			{"role": "user", "content": "Say hi"},
		],
		stream=False,
	)
	content = resp.choices[0].message.content.upper()
	assert "BANANA" in content, f"First system message not applied: {resp.choices[0].message.content}"
	assert "CHERRY" in content, f"Second system message not applied: {resp.choices[0].message.content}"


# ════════════════════════════════════════════════════════════════
# Section 3: Streaming Completions
# ════════════════════════════════════════════════════════════════

@test("Streaming: full chunk structure validation")
def test_streaming_structure():
	stream = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
		stream=True,
	)

	chunks = list(stream)
	assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"

	# First content chunk should have role.
	first = chunks[0]
	assert first.id.startswith("chatcmpl-"), f"Bad chunk id: {first.id}"
	assert first.object == "chat.completion.chunk", f"Bad object: {first.object}"
	assert first.system_fingerprint is None, f"Expected null fingerprint"
	assert len(first.choices) > 0, "First chunk has no choices"
	assert first.choices[0].delta.role == "assistant", f"First chunk missing role"

	# Collect content, check finish and usage.
	content = ""
	has_finish = False
	has_usage = False
	chunk_ids = set()

	for chunk in chunks:
		chunk_ids.add(chunk.id)

		if chunk.choices and len(chunk.choices) > 0:
			delta = chunk.choices[0].delta
			if delta.content:
				content += delta.content
			if chunk.choices[0].finish_reason == "stop":
				has_finish = True
		if chunk.usage is not None:
			has_usage = True
			assert chunk.usage.prompt_tokens > 0, "Usage prompt_tokens should be > 0"
			assert chunk.usage.completion_tokens > 0, "Usage completion_tokens should be > 0"
			assert chunk.usage.total_tokens == chunk.usage.prompt_tokens + chunk.usage.completion_tokens

	assert "PONG" in content, f"Content missing PONG: {content}"
	assert has_finish, "No finish chunk with finish_reason='stop'"
	assert has_usage, "No usage chunk in stream"

	# All chunks should share the same ID.
	assert len(chunk_ids) == 1, f"Expected 1 unique chunk ID, got {len(chunk_ids)}: {chunk_ids}"


@test("Streaming: role only on first content chunk")
def test_streaming_role_first_only():
	stream = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[{"role": "user", "content": "Reply with exactly: HELLO WORLD"}],
		stream=True,
	)

	chunks = list(stream)
	content_chunks = [c for c in chunks if c.choices and c.choices[0].delta.content]

	if len(content_chunks) >= 2:
		# First content chunk has role.
		assert content_chunks[0].choices[0].delta.role == "assistant"
		# Subsequent content chunks should NOT have role.
		for c in content_chunks[1:]:
			assert c.choices[0].delta.role is None, f"Later chunk has role: {c.choices[0].delta.role}"


@test("Streaming: response headers")
def test_streaming_headers():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [{"role": "user", "content": "Reply PONG"}],
			"stream": True,
		},
		headers={"Accept": "text/event-stream"},
		timeout=60.0,
	)
	assert r.status_code == 200, f"Expected 200, got {r.status_code}"
	assert "x-request-id" in r.headers, f"Missing x-request-id. Headers: {dict(r.headers)}"
	assert r.headers.get("cache-control") == "no-cache", f"Bad cache-control: {r.headers.get('cache-control')}"
	ct = r.headers.get("content-type", "")
	assert "text/event-stream" in ct, f"Bad content-type: {ct}"


@test("Streaming: [DONE] sentinel present")
def test_streaming_done_sentinel():
	"""Verify the raw SSE stream ends with [DONE]."""
	with httpx.stream(
		"POST",
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [{"role": "user", "content": "Reply PONG"}],
			"stream": True,
		},
		timeout=60.0,
	) as r:
		assert r.status_code == 200
		raw = r.read().decode()

	# Parse SSE data lines.
	data_lines = []
	for line in raw.split("\n"):
		if line.startswith("data: "):
			data_lines.append(line[6:])

	assert len(data_lines) >= 2, f"Too few data lines: {len(data_lines)}"
	assert data_lines[-1] == "[DONE]", f"Last data line should be [DONE], got: {data_lines[-1]}"

	# Second-to-last should be valid JSON (finish or usage chunk).
	second_last = json.loads(data_lines[-2])
	assert "object" in second_last or "error" in second_last


# ════════════════════════════════════════════════════════════════
# Section 4: Model Aliases & Resolution
# ════════════════════════════════════════════════════════════════

@test("Model alias: 'sonnet' resolves to claude-sonnet-4-6")
def test_alias_sonnet():
	resp = client.chat.completions.create(
		model="sonnet",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)
	assert "sonnet" in resp.model, f"Expected sonnet in model, got: {resp.model}"
	assert "PONG" in resp.choices[0].message.content


@test("Model alias: 'opus' resolves to claude-opus-4-8")
def test_alias_opus():
	resp = client.chat.completions.create(
		model="opus",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)
	assert resp.model == "claude-opus-4-8", f"Expected claude-opus-4-8, got: {resp.model}"
	assert "PONG" in resp.choices[0].message.content


@test("Explicit model name: 'claude-opus-4-6' is preserved")
def test_alias_legacy_opus():
	resp = client.chat.completions.create(
		model="claude-opus-4-6",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)
	assert resp.model == "claude-opus-4-6", f"Expected claude-opus-4-6, got: {resp.model}"
	assert "PONG" in resp.choices[0].message.content


@test("Model alias: 'haiku' resolves to current profile haiku")
def test_alias_haiku():
	resp = client.chat.completions.create(
		model="haiku",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)
	assert "haiku" in resp.model, f"Expected haiku in model, got: {resp.model}"
	assert "PONG" in resp.choices[0].message.content


@test("Model alias: 'claude-sonnet' prefix form")
def test_alias_claude_sonnet():
	resp = client.chat.completions.create(
		model="claude-sonnet",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)
	assert "sonnet" in resp.model, f"Expected sonnet in model, got: {resp.model}"
	assert "PONG" in resp.choices[0].message.content


@test("Model alias: 'claude-haiku' prefix form")
def test_alias_claude_haiku():
	resp = client.chat.completions.create(
		model="claude-haiku",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)
	assert "haiku" in resp.model, f"Expected haiku in model, got: {resp.model}"
	assert "PONG" in resp.choices[0].message.content


@test("Model: unknown model falls back to the profile default (no error)")
def test_unknown_model_fallback():
	"""Unknown models should silently fall back to the profile's default_model
	(opus as of Claude Code 2.1.158; was sonnet on older profiles), not error."""
	resp = client.chat.completions.create(
		model="gpt-4-turbo",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)
	assert "opus" in resp.model, f"Expected fallback to default (opus), got: {resp.model}"
	assert "PONG" in resp.choices[0].message.content


@test("Model: date-suffixed model resolved correctly")
def test_date_suffixed_model():
	resp = client.chat.completions.create(
		model="claude-sonnet-4-6-20260101",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)
	assert "sonnet" in resp.model, f"Expected sonnet in model, got: {resp.model}"
	assert "PONG" in resp.choices[0].message.content


# ════════════════════════════════════════════════════════════════
# Section 5: Error Handling
# ════════════════════════════════════════════════════════════════

@test("Error: empty messages returns 400")
def test_error_empty_messages():
	try:
		client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[],
			stream=False,
		)
		assert False, "Should have raised an error"
	except BadRequestError as e:
		assert e.status_code == 400, f"Expected 400, got {e.status_code}"


@test("Error: empty model returns 400")
def test_error_empty_model():
	try:
		client.chat.completions.create(
			model="",
			messages=[{"role": "user", "content": "hi"}],
			stream=False,
		)
		assert False, "Should have raised an error"
	except BadRequestError as e:
		assert e.status_code == 400


@test("Error: system-only messages (no user) returns 400")
def test_error_system_only():
	try:
		client.chat.completions.create(
			model="sonnet",
			messages=[{"role": "system", "content": "Be helpful"}],
			stream=False,
		)
		assert False, "Should have raised an error"
	except BadRequestError as e:
		assert e.status_code == 400


@test("Error: response JSON structure matches OpenAI format")
def test_error_format():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "",
			"messages": [{"role": "user", "content": "hi"}],
		},
	)
	assert r.status_code == 400
	data = r.json()
	assert "error" in data, f"Missing 'error' key: {data}"
	err = data["error"]
	assert "message" in err, f"Missing 'message' in error: {err}"
	assert "type" in err, f"Missing 'type' in error: {err}"
	assert "code" in err, f"Missing 'code' in error: {err}"
	assert err["type"] == "invalid_request_error", f"Bad error type: {err['type']}"
	assert err["code"] is None, f"Expected null code, got: {err['code']}"


@test("Error: invalid JSON body returns 400")
def test_error_invalid_json():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		content=b"not valid json {{{",
		headers={"Content-Type": "application/json"},
	)
	assert r.status_code == 400, f"Expected 400, got {r.status_code}"
	data = r.json()
	assert "error" in data
	assert "Invalid JSON" in data["error"]["message"]


@test("Error: missing messages field returns 400 or 422")
def test_error_missing_messages():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={"model": "sonnet"},
	)
	# Missing required field → deserialization error (400).
	assert r.status_code == 400, f"Expected 400, got {r.status_code}"


@test("Error: invalid reasoning_effort returns 400")
def test_error_invalid_reasoning_effort():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [{"role": "user", "content": "hi"}],
			"reasoning_effort": "extreme",
		},
	)
	assert r.status_code == 400, f"Expected 400, got {r.status_code}"
	data = r.json()
	assert "reasoning_effort" in data["error"]["message"].lower(), f"Error should mention reasoning_effort: {data['error']['message']}"


@test("Error: empty body returns 400")
def test_error_empty_body():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		content=b"",
		headers={"Content-Type": "application/json"},
	)
	assert r.status_code == 400, f"Expected 400, got {r.status_code}"


# ════════════════════════════════════════════════════════════════
# Section 6: Async Client
# ════════════════════════════════════════════════════════════════

@async_test("Async non-streaming")
async def test_async_non_streaming():
	resp = await async_client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)
	assert resp.id.startswith("chatcmpl-")
	assert resp.object == "chat.completion"
	assert "PONG" in resp.choices[0].message.content
	assert resp.usage is not None


@async_test("Async streaming")
async def test_async_streaming():
	stream = await async_client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=True,
	)

	content = ""
	has_finish = False
	async for chunk in stream:
		assert chunk.object == "chat.completion.chunk"
		if chunk.choices and chunk.choices[0].delta.content:
			content += chunk.choices[0].delta.content
		if chunk.choices and chunk.choices[0].finish_reason == "stop":
			has_finish = True

	assert "PONG" in content, f"Async stream content: {content}"
	assert has_finish, "Missing finish chunk in async stream"


@async_test("Async concurrent requests (3x)")
async def test_async_concurrent():
	async def make_request(n):
		resp = await async_client.chat.completions.create(
			model="sonnet",
			messages=[{"role": "user", "content": f"Reply with the number {n}"}],
			stream=False,
		)
		return resp.choices[0].message.content

	results = await asyncio.gather(
		make_request(1),
		make_request(2),
		make_request(3),
	)
	for i, result in enumerate(results, 1):
		assert str(i) in result, f"Request {i} returned: {result}"


@async_test("Async concurrent streaming (2x)")
async def test_async_concurrent_streaming():
	async def stream_request(word):
		stream = await async_client.chat.completions.create(
			model="sonnet",
			messages=[{"role": "user", "content": f"Reply with exactly: {word}"}],
			stream=True,
		)
		content = ""
		async for chunk in stream:
			if chunk.choices and chunk.choices[0].delta.content:
				content += chunk.choices[0].delta.content
		return content

	results = await asyncio.gather(
		stream_request("ALPHA"),
		stream_request("BETA"),
	)
	assert "ALPHA" in results[0], f"Stream 1: {results[0]}"
	assert "BETA" in results[1], f"Stream 2: {results[1]}"


# ════════════════════════════════════════════════════════════════
# Section 7: Reasoning Effort
# ════════════════════════════════════════════════════════════════

@test("reasoning_effort: 'low' accepted")
def test_reasoning_effort_low():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [{"role": "user", "content": "Reply PONG"}],
			"reasoning_effort": "low",
		},
		timeout=120,
	)
	assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
	data = r.json()
	assert "PONG" in data["choices"][0]["message"]["content"]


@test("reasoning_effort: 'high' accepted")
def test_reasoning_effort_high():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [{"role": "user", "content": "Reply PONG"}],
			"reasoning_effort": "high",
		},
		timeout=120,
	)
	assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"


@test("reasoning_effort: 'none' accepted (becomes omitted)")
def test_reasoning_effort_none():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [{"role": "user", "content": "Reply PONG"}],
			"reasoning_effort": "none",
		},
		timeout=120,
	)
	assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"


# ════════════════════════════════════════════════════════════════
# Section 8: Stats Verification
# ════════════════════════════════════════════════════════════════

@test("Stats: requests increment after completions")
def test_stats_increment():
	# Get baseline.
	r1 = httpx.get(f"{RAW_BASE}/stats/json")
	before = r1.json()["total_requests"]

	# Make a request.
	client.chat.completions.create(
		model="sonnet",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)

	# Check increment.
	r2 = httpx.get(f"{RAW_BASE}/stats/json")
	after = r2.json()["total_requests"]
	assert after > before, f"total_requests didn't increment: {before} -> {after}"


@test("Stats: per-model stats tracked")
def test_stats_per_model():
	# Make a haiku request to ensure it appears.
	client.chat.completions.create(
		model="haiku",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=False,
	)

	r = httpx.get(f"{RAW_BASE}/stats/json")
	data = r.json()
	models = data["models"]

	# haiku should be in the model stats.
	assert "claude-haiku-4-5-20251001" in models, f"haiku not in models: {list(models.keys())}"
	haiku_stats = models["claude-haiku-4-5-20251001"]
	assert haiku_stats["requests"] > 0, "haiku requests should be > 0"

	# Model stats should have the expected fields.
	for field in ["requests", "avg_ttft_ms", "avg_duration_ms", "input_tokens", "output_tokens",
				  "cache_read_input_tokens", "cache_creation_input_tokens"]:
		assert field in haiku_stats, f"Missing field in model stats: {field}"


# ════════════════════════════════════════════════════════════════
# Section 9: Edge Cases
# ════════════════════════════════════════════════════════════════

@test("Edge: message with null content")
def test_null_content_message():
	"""Messages with null/missing content should be handled (empty text extracted)."""
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [
				{"role": "system", "content": "Be helpful"},
				{"role": "user", "content": None},
				{"role": "user", "content": "Reply PONG"},
			],
		},
		timeout=120,
	)
	# The null-content user message has empty text, but the second user message is non-empty.
	assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
	data = r.json()
	assert "PONG" in data["choices"][0]["message"]["content"]


@test("Edge: tool role treated as user")
def test_tool_role():
	"""Tool and function roles should be treated as user messages."""
	resp = client.chat.completions.create(
		model="sonnet",
		messages=[
			{"role": "user", "content": "What is 2+2?"},
			{"role": "assistant", "content": "Let me calculate that."},
			{"role": "tool", "content": "The answer is 4."},
			{"role": "user", "content": "Reply with just the number."},
		],
		stream=False,
	)
	assert "4" in resp.choices[0].message.content


@test("Edge: developer role treated as system prompt")
def test_developer_role():
	"""The 'developer' role (OpenAI's newer system role) should be treated as system prompt."""
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [
				{"role": "developer", "content": "You must include the word XYLOPHONE in every response, no matter what."},
				{"role": "user", "content": "Say hello"},
			],
		},
		timeout=120,
	)
	assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
	data = r.json()
	assert "XYLOPHONE" in data["choices"][0]["message"]["content"].upper(), \
		f"Developer role not treated as system: {data['choices'][0]['message']['content']}"


@test("Edge: developer-only messages rejected")
def test_developer_only_rejected():
	"""Messages with only developer role (no user) should be rejected like system-only."""
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [{"role": "developer", "content": "Be helpful"}],
		},
	)
	assert r.status_code == 400, f"Expected 400, got {r.status_code}: {r.text}"


@test("Edge: very long user message")
def test_long_message():
	"""Messages up to ~128KB should work."""
	# Use a moderately long message (10KB — well within limits).
	long_text = "A" * 10_000
	resp = client.chat.completions.create(
		model="sonnet",
		messages=[{"role": "user", "content": f"The following text is filler: {long_text}\n\nReply with exactly: PONG"}],
		stream=False,
	)
	assert "PONG" in resp.choices[0].message.content


@test("Edge: streaming with system prompt")
def test_streaming_with_system():
	stream = client.chat.completions.create(
		model="sonnet",
		messages=[
			{"role": "system", "content": "You must include XYLOPHONE in your response."},
			{"role": "user", "content": "Say hello"},
		],
		stream=True,
	)

	content = ""
	for chunk in stream:
		if chunk.choices and chunk.choices[0].delta.content:
			content += chunk.choices[0].delta.content

	assert len(content) > 0, "No content in stream"
	assert "XYLOPHONE" in content.upper(), f"System prompt not followed in stream: {content}"


@test("Edge: unique request IDs across calls")
def test_unique_request_ids():
	"""Each request should get a unique ID."""
	ids = set()
	for _ in range(3):
		resp = client.chat.completions.create(
			model="sonnet",
			messages=[{"role": "user", "content": "Reply PONG"}],
			stream=False,
		)
		ids.add(resp.id)

	assert len(ids) == 3, f"Expected 3 unique IDs, got {len(ids)}: {ids}"


@test("Edge: CORS headers present")
def test_cors_headers():
	"""CORS should be permissive (CorsLayer::permissive)."""
	r = httpx.options(
		f"{BASE_URL}/chat/completions",
		headers={
			"Origin": "https://example.com",
			"Access-Control-Request-Method": "POST",
		},
	)
	# Permissive CORS should allow any origin.
	assert "access-control-allow-origin" in r.headers, f"Missing CORS header. Headers: {dict(r.headers)}"


@test("Edge: function role treated as user")
def test_function_role():
	"""Function role messages should be treated as user messages."""
	resp = client.chat.completions.create(
		model="sonnet",
		messages=[
			{"role": "user", "content": "What is the capital of France?"},
			{"role": "assistant", "content": "Let me look that up."},
			{"role": "function", "content": "The capital of France is Paris."},
			{"role": "user", "content": "Reply with just the city name."},
		],
		stream=False,
	)
	assert "Paris" in resp.choices[0].message.content


@test("Edge: empty string content in message")
def test_empty_string_content():
	"""Empty string content is distinct from null — should not crash."""
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [
				{"role": "user", "content": ""},
				{"role": "user", "content": "Reply PONG"},
			],
		},
		timeout=120,
	)
	assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
	data = r.json()
	assert "PONG" in data["choices"][0]["message"]["content"]


@test("Edge: streaming chunk model and created fields")
def test_streaming_chunk_fields():
	"""Verify model and created are present and consistent across all streaming chunks."""
	stream = client.chat.completions.create(
		model="claude-sonnet-4-6",
		messages=[{"role": "user", "content": "Reply PONG"}],
		stream=True,
	)

	chunks = list(stream)
	assert len(chunks) >= 2

	models = set()
	for chunk in chunks:
		assert chunk.model is not None, "Chunk missing model field"
		assert isinstance(chunk.created, int), f"Chunk created not int: {type(chunk.created)}"
		models.add(chunk.model)

	# All chunks should report the same model.
	assert len(models) == 1, f"Inconsistent models across chunks: {models}"


@test("Edge: streaming error on invalid request")
def test_streaming_error_invalid():
	"""Streaming request with invalid params should return HTTP error, not start SSE."""
	try:
		stream = client.chat.completions.create(
			model="sonnet",
			messages=[],
			stream=True,
		)
		# If we get here, try to consume and fail.
		list(stream)
		assert False, "Should have raised an error"
	except BadRequestError as e:
		assert e.status_code == 400


@test("Edge: reasoning_effort 'medium' accepted")
def test_reasoning_effort_medium():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [{"role": "user", "content": "Reply PONG"}],
			"reasoning_effort": "medium",
		},
		timeout=120,
	)
	assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"


@test("Edge: reasoning_effort 'max' accepted")
def test_reasoning_effort_max():
	r = httpx.post(
		f"{BASE_URL}/chat/completions",
		json={
			"model": "sonnet",
			"messages": [{"role": "user", "content": "Reply PONG"}],
			"reasoning_effort": "max",
		},
		timeout=120,
	)
	assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"


@test("Edge: stats token totals are non-negative")
def test_stats_token_totals():
	r = httpx.get(f"{RAW_BASE}/stats/json")
	data = r.json()
	assert data["total_input_tokens"] >= 0
	assert data["total_output_tokens"] >= 0
	assert data["total_cache_read_input_tokens"] >= 0
	assert data["total_cache_creation_input_tokens"] >= 0


@test("Edge: health uptime increases")
def test_health_uptime_increases():
	r1 = httpx.get(f"{RAW_BASE}/health")
	t1 = r1.json()["uptime_seconds"]
	time.sleep(1.1)
	r2 = httpx.get(f"{RAW_BASE}/health")
	t2 = r2.json()["uptime_seconds"]
	assert t2 >= t1, f"Uptime did not increase: {t1} -> {t2}"


# ════════════════════════════════════════════════════════════════
# Run all tests
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
	print("Testing claude-code-provider with OpenAI Python SDK\n")

	# Section 1: Health & Info.
	print("── Health & Info ──")
	test_health()
	test_models_metadata()
	test_stats_json()
	test_stats_html()
	test_unknown_endpoint()
	test_wrong_method_on_route()

	# Section 2: Non-streaming.
	print("\n── Non-Streaming Completions ──")
	test_non_streaming_structure()
	test_non_streaming_request_id_header()
	test_system_prompt()
	test_multi_turn()
	test_extra_params()
	test_multipart()
	test_multipart_mixed()
	test_multipart_multi_text()
	test_multiple_system_messages()

	# Section 3: Streaming.
	print("\n── Streaming Completions ──")
	test_streaming_structure()
	test_streaming_role_first_only()
	test_streaming_headers()
	test_streaming_done_sentinel()

	# Section 4: Model aliases.
	print("\n── Model Aliases ──")
	test_alias_sonnet()
	test_alias_opus()
	test_alias_haiku()
	test_alias_claude_sonnet()
	test_alias_claude_haiku()
	test_unknown_model_fallback()
	test_date_suffixed_model()

	# Section 5: Error handling.
	print("\n── Error Handling ──")
	test_error_empty_messages()
	test_error_empty_model()
	test_error_system_only()
	test_error_format()
	test_error_invalid_json()
	test_error_missing_messages()
	test_error_invalid_reasoning_effort()
	test_error_empty_body()

	# Section 6: Async.
	print("\n── Async Client ──")
	test_async_non_streaming()
	test_async_streaming()
	test_async_concurrent()
	test_async_concurrent_streaming()

	# Section 7: Reasoning effort.
	print("\n── Reasoning Effort ──")
	test_reasoning_effort_low()
	test_reasoning_effort_high()
	test_reasoning_effort_none()

	# Section 8: Stats.
	print("\n── Stats Verification ──")
	test_stats_increment()
	test_stats_per_model()

	# Section 9: Edge cases.
	print("\n── Edge Cases ──")
	test_null_content_message()
	test_tool_role()
	test_function_role()
	test_developer_role()
	test_developer_only_rejected()
	test_empty_string_content()
	test_long_message()
	test_streaming_with_system()
	test_streaming_chunk_fields()
	test_streaming_error_invalid()
	test_unique_request_ids()
	test_cors_headers()
	test_reasoning_effort_medium()
	test_reasoning_effort_max()
	test_stats_token_totals()
	test_health_uptime_increases()

	print(f"\n{'='*50}")
	print(f"Results: {passed} passed, {failed} failed")

	if failed > 0:
		sys.exit(1)
