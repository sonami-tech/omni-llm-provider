"""
Comprehensive integration tests for claude-code-provider.

Tests are organized into classes by feature area. All tests run against
a live CCP server that is automatically built and started by conftest.py.

Run with: ./tests/run.sh
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
#     "openai>=1.0",
#     "pytest>=8.0",
#     "pytest-asyncio>=0.24",
#     "xxhash>=3.5",
# ]
# ///

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

import httpx
import pytest
import xxhash
from openai import AsyncOpenAI, BadRequestError, OpenAI


DEFAULT_PROFILE = {
	"name": "cc-2.1.161-sdk-cli",
	"version": "2.1.161",
	"say_ok_suffix": "d2b",
}
CLAUDE_CODE_PREAMBLE = "You are Claude Code, Anthropic's official CLI for Claude."
CCH_SEED = 0x4D659218E32A3268
CCH_RE = re.compile(r"(cc_entrypoint=sdk-cli; cch=)([0-9a-f]{5})(;)")


def _extract_logged_anthropic_requests(log_file):
	# Conversation logs write the serialized JSON body followed by a dashed
	# separator line; this parser deliberately keys off that separator so
	# braces inside JSON strings do not terminate a match early.
	content = open(log_file).read()
	requests = []
	for match in re.finditer(r"\n(\{.*?\})\n-+\n", content, re.DOTALL):
		raw = match.group(1)
		try:
			obj = json.loads(raw)
		except json.JSONDecodeError:
			continue
		if "messages" in obj and "system" in obj:
			requests.append((obj, raw))
	return requests


def _anthropic_request_matches_profile(req, profile, expected_suffix=None):
	system = req["system"]
	if not isinstance(system, list):
		return False
	if len(system) < 2:
		return False
	header = system[0].get("text")
	return (
		_billing_header_matches_profile(header, profile, expected_suffix)
		and system[1].get("text") == CLAUDE_CODE_PREAMBLE
	)


def _billing_header_matches_profile(header, profile, expected_suffix=None):
	if not isinstance(header, str):
		return False
	if expected_suffix is None:
		expected_suffix = re.escape(profile["say_ok_suffix"])
	elif expected_suffix == "*":
		expected_suffix = r"[0-9a-f]{3}"
	else:
		expected_suffix = re.escape(expected_suffix)
	return re.fullmatch(
		rf"x-anthropic-billing-header: cc_version={re.escape(profile['version'])}\."
		rf"{expected_suffix}; cc_entrypoint=sdk-cli; cch=[0-9a-f]{{5}};",
		header,
	) is not None


def _assert_logged_request_cch_is_self_consistent(req, raw_body, profile, expected_suffix=None):
	system = req["system"]
	header = system[0]["text"]
	header_match = CCH_RE.search(header)
	assert header_match is not None, header
	actual = header_match.group(2)
	assert actual != "00000"

	placeholder_header = CCH_RE.sub(r"\g<1>00000\g<3>", header, count=1)
	assert placeholder_header != header
	placeholder = raw_body.replace(header, placeholder_header, 1).encode()
	expected = f"{xxhash.xxh64(placeholder, seed=CCH_SEED).intdigest() & 0xfffff:05x}"
	assert actual == expected
	assert _anthropic_request_matches_profile(req, profile, expected_suffix)


def _logged_anthropic_request_count(log_file):
	if not os.path.exists(log_file):
		return 0
	return len(_extract_logged_anthropic_requests(log_file))


def _wait_for_new_logged_anthropic_request(log_file, previous_count):
	deadline = time.monotonic() + 10
	while time.monotonic() < deadline:
		requests = _extract_logged_anthropic_requests(log_file)
		if len(requests) > previous_count:
			return requests[-1]
		time.sleep(0.2)
	raise AssertionError(
		f"No new Anthropic request found in {log_file}; "
		f"previous_count={previous_count}"
	)


def _assert_profile_identity(log_file, profile, previous_count):
	req, raw_body = _wait_for_new_logged_anthropic_request(log_file, previous_count)
	system = req["system"]
	assert isinstance(system, list)
	_assert_logged_request_cch_is_self_consistent(req, raw_body, profile)


def _wait_for_logged_profile_request(log_file, profile, previous_count, expected_suffix=None):
	deadline = time.monotonic() + 10
	last_seen = previous_count
	while time.monotonic() < deadline:
		requests = _extract_logged_anthropic_requests(log_file)
		last_seen = len(requests)
		for req, raw_body in requests[previous_count:]:
			if _anthropic_request_matches_profile(req, profile, expected_suffix):
				_assert_logged_request_cch_is_self_consistent(
					req,
					raw_body,
					profile,
					expected_suffix,
				)
				return req, raw_body
		time.sleep(0.2)
	raise AssertionError(
		f"No new Anthropic request with profile {profile['name']} found in "
		f"{log_file}; previous_count={previous_count}; last_seen={last_seen}"
	)


# ════════════════════════════════════════════════════════════════
# Health & Info Endpoints
# ════════════════════════════════════════════════════════════════


class TestHealth:
	def test_health_structure(self, base_url):
		r = httpx.get(f"{base_url}/health")
		assert r.status_code == 200
		data = r.json()
		assert data["status"] == "ok"
		assert isinstance(data["uptime_seconds"], int)
		assert data["uptime_seconds"] >= 0
		assert isinstance(data["active_requests"], int)

	def test_health_uptime_increases(self, base_url):
		r1 = httpx.get(f"{base_url}/health")
		t1 = r1.json()["uptime_seconds"]
		time.sleep(1.1)
		r2 = httpx.get(f"{base_url}/health")
		t2 = r2.json()["uptime_seconds"]
		assert t2 >= t1

	def test_unknown_endpoint_returns_404(self, base_url):
		r = httpx.get(f"{base_url}/v1/nonexistent")
		assert r.status_code == 404
		data = r.json()
		assert "error" in data
		assert data["error"]["type"] == "invalid_request_error"
		assert data["error"]["code"] is None

	def test_wrong_method_returns_error(self, base_url):
		r = httpx.post(f"{base_url}/v1/models", content=b"{}")
		assert r.status_code in (404, 405)


class TestModels:
	def test_models_list_structure(self, client):
		models = client.models.list()
		assert hasattr(models, "data")
		assert len(models.data) == 3

		names = [m.id for m in models.data]
		assert "claude-opus-4-8" in names
		assert "claude-sonnet-4-6" in names
		assert "claude-haiku-4-5-20251001" in names

		for m in models.data:
			assert m.object == "model"
			assert m.owned_by == "anthropic"

	def test_models_extended_metadata(self, api_base_url):
		r = httpx.get(f"{api_base_url}/models")
		data = r.json()
		assert data["object"] == "list"

		by_id = {m["id"]: m for m in data["data"]}

		assert by_id["claude-opus-4-8"]["context_window"] == 1_000_000
		assert by_id["claude-opus-4-8"]["max_tokens"] == 128_000
		assert by_id["claude-sonnet-4-6"]["context_window"] == 1_000_000
		assert by_id["claude-sonnet-4-6"]["max_tokens"] == 64_000
		assert by_id["claude-haiku-4-5-20251001"]["context_window"] == 200_000
		assert by_id["claude-haiku-4-5-20251001"]["max_tokens"] == 64_000

		for m in data["data"]:
			assert m["created"] == 0


class TestStatsEndpoints:
	def test_stats_json_structure(self, base_url):
		r = httpx.get(f"{base_url}/stats/json")
		assert r.status_code == 200
		data = r.json()

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

	def test_stats_html_response(self, base_url):
		r = httpx.get(f"{base_url}/stats")
		assert r.status_code == 200
		assert "text/html" in r.headers.get("content-type", "")
		assert "<!DOCTYPE html>" in r.text
		assert "Claude Code Provider" in r.text

	def test_stats_token_totals_non_negative(self, base_url):
		r = httpx.get(f"{base_url}/stats/json")
		data = r.json()
		assert data["total_input_tokens"] >= 0
		assert data["total_output_tokens"] >= 0
		assert data["total_cache_read_input_tokens"] >= 0
		assert data["total_cache_creation_input_tokens"] >= 0


# ════════════════════════════════════════════════════════════════
# Non-Streaming Completions
# ════════════════════════════════════════════════════════════════


class TestNonStreaming:
	def test_full_response_structure(self, client):
		resp = client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
			stream=False,
		)
		assert resp.id.startswith("chatcmpl-")
		assert len(resp.id) == len("chatcmpl-") + 8
		assert resp.object == "chat.completion"
		assert isinstance(resp.created, int)
		assert abs(resp.created - int(time.time())) < 120
		assert resp.model is not None
		assert resp.system_fingerprint is None or isinstance(resp.system_fingerprint, str)
		assert len(resp.choices) == 1

		choice = resp.choices[0]
		assert choice.index == 0
		assert choice.finish_reason == "stop"
		assert choice.message.role == "assistant"
		assert "PONG" in choice.message.content

		assert resp.usage is not None
		assert resp.usage.prompt_tokens > 0
		assert resp.usage.completion_tokens > 0
		assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens

	def test_request_id_header(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "Reply PONG"}],
				"stream": False,
			},
			timeout=120,
		)
		assert r.status_code == 200
		assert "x-request-id" in r.headers
		req_id = r.headers["x-request-id"]
		assert len(req_id) == 8
		assert r.json()["id"] == f"chatcmpl-{req_id}"

	def test_system_prompt_influences_response(self, client):
		resp = client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[
				{"role": "system", "content": "You must include the word XYLOPHONE in every response, no matter what."},
				{"role": "user", "content": "Say hello"},
			],
			stream=False,
		)
		assert "XYLOPHONE" in resp.choices[0].message.content.upper()

	def test_multi_turn_conversation(self, client):
		resp = client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[
				{"role": "user", "content": "My name is Alice."},
				{"role": "assistant", "content": "Hello Alice!"},
				{"role": "user", "content": "What is my name? Reply with just the name."},
			],
			stream=False,
		)
		assert "Alice" in resp.choices[0].message.content

	def test_extra_openai_params_accepted(self, client):
		resp = client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[{"role": "user", "content": "Reply PONG"}],
			stream=False,
			max_tokens=100,
			temperature=0.5,
			top_p=0.9,
		)
		assert "PONG" in resp.choices[0].message.content

	def test_multipart_content_array(self, client):
		resp = client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[{
				"role": "user",
				"content": [{"type": "text", "text": "Reply with exactly: PONG"}],
			}],
			stream=False,
		)
		assert "PONG" in resp.choices[0].message.content

	def test_multipart_mixed_types(self, client):
		"""image_url parts are translated to Anthropic image blocks (vision
		passthrough), not dropped. Uses an inline base64 data URL so the test is
		hermetic (no external image fetch) and exercises the base64 path."""
		# A real 16x16 solid-red PNG (a 1x1 image is rejected by the vision model).
		png_b64 = (
			"iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAFklEQVR42mO4"
			"IydHEmIY1TCqYfhqAACaMxgQdrf9VwAAAABJRU5ErkJggg=="
		)
		resp = client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[{
				"role": "user",
				"content": [
					{"type": "text", "text": "Reply with exactly: PONG"},
					{
						"type": "image_url",
						"image_url": {"url": f"data:image/png;base64,{png_b64}"},
					},
				],
			}],
			stream=False,
		)
		# A successful completion proves the image block was accepted by Anthropic
		# (a dropped image would still answer; a malformed one would 400).
		assert "PONG" in resp.choices[0].message.content

	def test_multiple_text_parts_concatenated(self, client):
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

	def test_multiple_system_messages_concatenated(self, client):
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
		assert "BANANA" in content
		assert "CHERRY" in content

	def test_unique_request_ids(self, client):
		ids = set()
		for _ in range(3):
			resp = client.chat.completions.create(
				model="sonnet",
				messages=[{"role": "user", "content": "Reply PONG"}],
				stream=False,
			)
			ids.add(resp.id)
		assert len(ids) == 3


# ════════════════════════════════════════════════════════════════
# Streaming Completions
# ════════════════════════════════════════════════════════════════


class TestStreaming:
	def test_full_chunk_structure(self, client):
		stream = client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
			stream=True,
		)
		chunks = list(stream)
		assert len(chunks) >= 2

		first = chunks[0]
		assert first.id.startswith("chatcmpl-")
		assert first.object == "chat.completion.chunk"
		assert first.system_fingerprint is None
		assert len(first.choices) > 0
		assert first.choices[0].delta.role == "assistant"

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
				assert chunk.usage.prompt_tokens > 0
				assert chunk.usage.completion_tokens > 0
				assert chunk.usage.total_tokens == chunk.usage.prompt_tokens + chunk.usage.completion_tokens

		assert "PONG" in content
		assert has_finish
		assert has_usage
		assert len(chunk_ids) == 1

	def test_role_only_on_first_content_chunk(self, client):
		stream = client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[{"role": "user", "content": "Reply with exactly: HELLO WORLD"}],
			stream=True,
		)
		chunks = list(stream)
		content_chunks = [c for c in chunks if c.choices and c.choices[0].delta.content]

		if len(content_chunks) >= 2:
			assert content_chunks[0].choices[0].delta.role == "assistant"
			for c in content_chunks[1:]:
				assert c.choices[0].delta.role is None

	def test_response_headers(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "Reply PONG"}],
				"stream": True,
			},
			headers={"Accept": "text/event-stream"},
			timeout=60.0,
		)
		assert r.status_code == 200
		assert "x-request-id" in r.headers
		assert r.headers.get("cache-control") == "no-cache"
		assert "text/event-stream" in r.headers.get("content-type", "")

	def test_done_sentinel_present(self, api_base_url):
		with httpx.stream(
			"POST",
			f"{api_base_url}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "Reply PONG"}],
				"stream": True,
			},
			timeout=60.0,
		) as r:
			assert r.status_code == 200
			raw = r.read().decode()

		data_lines = [line[6:] for line in raw.split("\n") if line.startswith("data: ")]
		assert len(data_lines) >= 2
		assert data_lines[-1] == "[DONE]"

		second_last = json.loads(data_lines[-2])
		assert "object" in second_last or "error" in second_last

	def test_streaming_with_system_prompt(self, client):
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
		assert len(content) > 0
		assert "XYLOPHONE" in content.upper()

	def test_streaming_chunk_model_and_created_consistent(self, client):
		stream = client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[{"role": "user", "content": "Reply PONG"}],
			stream=True,
		)
		chunks = list(stream)
		assert len(chunks) >= 2

		models = set()
		for chunk in chunks:
			assert chunk.model is not None
			assert isinstance(chunk.created, int)
			models.add(chunk.model)
		assert len(models) == 1

	def test_streaming_error_on_invalid_request(self, client):
		with pytest.raises(BadRequestError) as exc_info:
			stream = client.chat.completions.create(
				model="sonnet",
				messages=[],
				stream=True,
			)
			list(stream)
		assert exc_info.value.status_code == 400


# ════════════════════════════════════════════════════════════════
# Model Aliases & Resolution
# ════════════════════════════════════════════════════════════════


class TestModelAliases:
	@pytest.mark.parametrize("alias,expected_substr", [
		("sonnet", "sonnet"),
		("opus", "opus"),
		("haiku", "haiku"),
		("claude-opus-4-6", "opus"),
		("claude-opus-4-6-20260101", "opus"),
		("claude-sonnet", "sonnet"),
		("claude-haiku", "haiku"),
		("claude-sonnet-4-6-20260101", "sonnet"),
	])
	def test_alias_resolves(self, client, alias, expected_substr):
		resp = client.chat.completions.create(
			model=alias,
			messages=[{"role": "user", "content": "Reply PONG"}],
			stream=False,
		)
		assert expected_substr in resp.model
		assert "PONG" in resp.choices[0].message.content

	def test_opus_alias_returns_profile_canonical_model(self, client):
		resp = client.chat.completions.create(
			model="opus",
			messages=[{"role": "user", "content": "Reply PONG"}],
			stream=False,
		)
		assert resp.model == "claude-opus-4-8"

	def test_explicit_claude_model_is_preserved(self, client):
		resp = client.chat.completions.create(
			model="claude-opus-4-6",
			messages=[{"role": "user", "content": "Reply PONG"}],
			stream=False,
		)
		assert resp.model == "claude-opus-4-6"

	def test_unknown_model_falls_back_to_default(self, client):
		resp = client.chat.completions.create(
			model="gpt-4-turbo",
			messages=[{"role": "user", "content": "Reply PONG"}],
			stream=False,
		)
		# 2.1.161 default_model is opus (captured 2026-06-03); older profiles used sonnet.
		assert "opus" in resp.model
		assert "PONG" in resp.choices[0].message.content


# ════════════════════════════════════════════════════════════════
# Error Handling
# ════════════════════════════════════════════════════════════════


class TestErrors:
	def test_empty_messages_400(self, client):
		with pytest.raises(BadRequestError) as exc_info:
			client.chat.completions.create(
				model="claude-sonnet-4-6",
				messages=[],
				stream=False,
			)
		assert exc_info.value.status_code == 400

	def test_empty_model_400(self, client):
		with pytest.raises(BadRequestError) as exc_info:
			client.chat.completions.create(
				model="",
				messages=[{"role": "user", "content": "hi"}],
				stream=False,
			)
		assert exc_info.value.status_code == 400

	def test_system_only_messages_400(self, client):
		with pytest.raises(BadRequestError) as exc_info:
			client.chat.completions.create(
				model="sonnet",
				messages=[{"role": "system", "content": "Be helpful"}],
				stream=False,
			)
		assert exc_info.value.status_code == 400

	def test_error_json_format_matches_openai(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={"model": "", "messages": [{"role": "user", "content": "hi"}]},
		)
		assert r.status_code == 400
		err = r.json()["error"]
		assert "message" in err
		assert "type" in err
		assert "code" in err
		assert err["type"] == "invalid_request_error"
		assert err["code"] is None

	def test_invalid_json_body_400(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			content=b"not valid json {{{",
			headers={"Content-Type": "application/json"},
		)
		assert r.status_code == 400
		assert "Invalid JSON" in r.json()["error"]["message"]

	def test_missing_messages_field_400(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={"model": "sonnet"},
		)
		assert r.status_code == 400

	def test_invalid_reasoning_effort_400(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "hi"}],
				"reasoning_effort": "extreme",
			},
		)
		assert r.status_code == 400
		assert "reasoning_effort" in r.json()["error"]["message"].lower()

	def test_empty_body_400(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			content=b"",
			headers={"Content-Type": "application/json"},
		)
		assert r.status_code == 400

	def test_developer_only_messages_400(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "developer", "content": "Be helpful"}],
			},
		)
		assert r.status_code == 400


# ════════════════════════════════════════════════════════════════
# Reasoning Effort
# ════════════════════════════════════════════════════════════════


class TestReasoningEffort:
	@pytest.mark.parametrize("effort", ["low", "medium", "high", "max", "none"])
	def test_valid_effort_accepted(self, api_base_url, effort):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "Reply PONG"}],
				"reasoning_effort": effort,
			},
			timeout=120,
		)
		assert r.status_code == 200


# ════════════════════════════════════════════════════════════════
# Edge Cases
# ════════════════════════════════════════════════════════════════


class TestEdgeCases:
	def test_null_content_message(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
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
		assert r.status_code == 200
		assert "PONG" in r.json()["choices"][0]["message"]["content"]

	def test_tool_role_as_user(self, client):
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

	def test_developer_role_as_system(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [
					{"role": "developer", "content": "You must include the word XYLOPHONE in every response, no matter what."},
					{"role": "user", "content": "Say hello"},
				],
			},
			timeout=120,
		)
		assert r.status_code == 200
		assert "XYLOPHONE" in r.json()["choices"][0]["message"]["content"].upper()

	def test_function_role_as_user(self, client):
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

	def test_empty_string_content(self, api_base_url):
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [
					{"role": "user", "content": ""},
					{"role": "user", "content": "Reply PONG"},
				],
			},
			timeout=120,
		)
		assert r.status_code == 200
		assert "PONG" in r.json()["choices"][0]["message"]["content"]

	def test_cors_headers(self, api_base_url):
		r = httpx.options(
			f"{api_base_url}/chat/completions",
			headers={
				"Origin": "https://example.com",
				"Access-Control-Request-Method": "POST",
			},
		)
		assert "access-control-allow-origin" in r.headers

	def test_both_path_variants_work(self, base_url):
		"""Both /v1/chat/completions and /chat/completions should work."""
		payload = {
			"model": "sonnet",
			"messages": [{"role": "user", "content": "Reply PONG"}],
		}
		r1 = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
		r2 = httpx.post(f"{base_url}/chat/completions", json=payload, timeout=120)
		assert r1.status_code == 200
		assert r2.status_code == 200

	def test_both_models_path_variants(self, base_url):
		"""Both /v1/models and /models should work."""
		r1 = httpx.get(f"{base_url}/v1/models")
		r2 = httpx.get(f"{base_url}/models")
		assert r1.status_code == 200
		assert r2.status_code == 200
		assert r1.json() == r2.json()


# ════════════════════════════════════════════════════════════════
# Tool Calling
# ════════════════════════════════════════════════════════════════


class TestToolCalling:
	"""Tool/function call passthrough end-to-end."""

	@staticmethod
	def _tools():
		return [
			{
				"type": "function",
				"function": {
					"name": "search",
					"description": "Search files matching a glob pattern.",
					"parameters": {
						"type": "object",
						"properties": {"pattern": {"type": "string"}},
						"required": ["pattern"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "read",
					"description": "Read the contents of a file.",
					"parameters": {
						"type": "object",
						"properties": {"path": {"type": "string"}},
						"required": ["path"],
					},
				},
			},
		]

	def test_first_turn_emits_tool_call(self, api_base_url):
		"""Initial tool request should produce a tool_calls response."""
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={
				"model": "claude-haiku-4-5",
				"messages": [{"role": "user", "content": "Search for *.rs files."}],
				"tools": self._tools(),
			},
			timeout=120,
		)
		assert r.status_code == 200, r.text
		choice = r.json()["choices"][0]
		assert choice["message"].get("tool_calls"), choice
		assert choice["message"]["tool_calls"][0]["function"]["name"] == "search"

	def test_multi_turn_after_tool_result(self, api_base_url):
		"""Regression: multi-turn conversations with prior tool_calls + tool result.
		Previously caused 'tool call could not be parsed (retry also failed)' or hung
		due to the CLI's agentic retry loop. After replacing the CLI's default system
		prompt with the CCP preamble, this completes without error."""
		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={
				"model": "claude-haiku-4-5",
				"messages": [
					{"role": "user", "content": "Search for *.rs files."},
					{
						"role": "assistant",
						"tool_calls": [
							{
								"id": "call_1",
								"type": "function",
								"function": {
									"name": "search",
									"arguments": '{"pattern":"*.rs"}',
								},
							}
						],
					},
					{
						"role": "tool",
						"tool_call_id": "call_1",
						"content": "Found: main.rs, lib.rs, config.rs",
					},
					{"role": "user", "content": "Now read main.rs."},
				],
				"tools": self._tools(),
			},
			timeout=120,
		)
		assert r.status_code == 200, r.text
		body = r.json()
		# Response must succeed (no error). The model may either emit a tool_call
		# for `read` or fall back to a text response — both are acceptable. The
		# regression we are guarding against is an empty error or 5xx.
		choice = body["choices"][0]
		message = choice["message"]
		assert message.get("tool_calls") or message.get("content"), body


# ════════════════════════════════════════════════════════════════
# Async Client
# ════════════════════════════════════════════════════════════════


class TestAsync:
	@pytest.mark.asyncio
	async def test_async_non_streaming(self, async_client):
		resp = await async_client.chat.completions.create(
			model="claude-sonnet-4-6",
			messages=[{"role": "user", "content": "Reply PONG"}],
			stream=False,
		)
		assert resp.id.startswith("chatcmpl-")
		assert resp.object == "chat.completion"
		assert "PONG" in resp.choices[0].message.content
		assert resp.usage is not None

	@pytest.mark.asyncio
	async def test_async_streaming(self, async_client):
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
		assert "PONG" in content
		assert has_finish

	@pytest.mark.asyncio
	async def test_concurrent_requests(self, async_client):
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
			assert str(i) in result

	@pytest.mark.asyncio
	async def test_concurrent_streaming(self, async_client):
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
		assert "ALPHA" in results[0]
		assert "BETA" in results[1]


# ════════════════════════════════════════════════════════════════
# Stats Tracking
# ════════════════════════════════════════════════════════════════


class TestStats:
	def test_requests_increment(self, client, base_url):
		before = httpx.get(f"{base_url}/stats/json").json()["total_requests"]
		client.chat.completions.create(
			model="sonnet",
			messages=[{"role": "user", "content": "Reply PONG"}],
			stream=False,
		)
		after = httpx.get(f"{base_url}/stats/json").json()["total_requests"]
		assert after > before

	def test_per_model_stats(self, client, base_url):
		client.chat.completions.create(
			model="haiku",
			messages=[{"role": "user", "content": "Reply PONG"}],
			stream=False,
		)
		data = httpx.get(f"{base_url}/stats/json").json()
		models = data["models"]
		assert "claude-haiku-4-5-20251001" in models
		haiku = models["claude-haiku-4-5-20251001"]
		assert haiku["requests"] > 0

		for field in [
			"requests", "avg_ttft_ms", "avg_duration_ms",
			"input_tokens", "output_tokens",
			"cache_read_input_tokens", "cache_creation_input_tokens",
		]:
			assert field in haiku


# ════════════════════════════════════════════════════════════════
# Authentication (uses separate auth-enabled server)
# ════════════════════════════════════════════════════════════════


class TestAuth:
	def test_valid_key_accepted(self, ccp_auth_server):
		r = httpx.post(
			f"{ccp_auth_server['api_base_url']}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "Reply PONG"}],
			},
			headers={"Authorization": f"Bearer {ccp_auth_server['api_key']}"},
			timeout=120,
		)
		assert r.status_code == 200
		assert "PONG" in r.json()["choices"][0]["message"]["content"]

	def test_invalid_key_rejected(self, ccp_auth_server):
		r = httpx.post(
			f"{ccp_auth_server['api_base_url']}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "Reply PONG"}],
			},
			headers={"Authorization": "Bearer wrong-key-1234567890"},
			timeout=10,
		)
		assert r.status_code == 401
		assert "error" in r.json()
		assert "Invalid API key" in r.json()["error"]["message"]

	def test_missing_key_rejected(self, ccp_auth_server):
		r = httpx.post(
			f"{ccp_auth_server['api_base_url']}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "Reply PONG"}],
			},
			timeout=10,
		)
		assert r.status_code == 401
		assert "Missing API key" in r.json()["error"]["message"]

	def test_health_bypasses_auth(self, ccp_auth_server):
		"""Health endpoint should be accessible without auth."""
		r = httpx.get(f"{ccp_auth_server['base_url']}/health")
		assert r.status_code == 200
		assert r.json()["status"] == "ok"

	def test_stats_requires_auth(self, ccp_auth_server):
		"""Stats endpoints sit behind the same auth layer as the rest of the API:
		when API auth is enabled they require a key (no anonymous key-id/usage
		exposure), and they succeed with a valid key."""
		# Without a key: rejected, same as the other API routes.
		r = httpx.get(f"{ccp_auth_server['base_url']}/stats/json")
		assert r.status_code == 401
		# With a valid key: accessible.
		r = httpx.get(
			f"{ccp_auth_server['base_url']}/stats/json",
			headers={"Authorization": f"Bearer {ccp_auth_server['api_key']}"},
		)
		assert r.status_code == 200
		assert "total_requests" in r.json()

	def test_models_requires_auth(self, ccp_auth_server):
		"""GET /v1/models is an API route and requires auth."""
		r = httpx.get(f"{ccp_auth_server['api_base_url']}/models")
		assert r.status_code == 401

	def test_models_with_auth(self, ccp_auth_server):
		r = httpx.get(
			f"{ccp_auth_server['api_base_url']}/models",
			headers={"Authorization": f"Bearer {ccp_auth_server['api_key']}"},
		)
		assert r.status_code == 200
		assert len(r.json()["data"]) == 3

	def test_per_key_stats_tracked(self, ccp_auth_server):
		"""Requests should be tracked per API key."""
		httpx.post(
			f"{ccp_auth_server['api_base_url']}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "Reply PONG"}],
			},
			headers={"Authorization": f"Bearer {ccp_auth_server['api_key']}"},
			timeout=120,
		)
		data = httpx.get(
			f"{ccp_auth_server['base_url']}/stats/json",
			headers={"Authorization": f"Bearer {ccp_auth_server['api_key']}"},
		).json()
		assert "api_keys" in data
		assert len(data["api_keys"]) > 0


# ════════════════════════════════════════════════════════════════
# Large Prompts
# ════════════════════════════════════════════════════════════════


class TestLargePrompts:
	def test_10kb_prompt(self, client):
		"""10KB prompt should work fine."""
		filler = "A" * 10_000
		resp = client.chat.completions.create(
			model="sonnet",
			messages=[{"role": "user", "content": f"Filler: {filler}\n\nReply with exactly: PONG"}],
			stream=False,
		)
		assert "PONG" in resp.choices[0].message.content

	def test_200kb_prompt_streaming(self, api_base_url):
		"""Large prompt should also work in streaming mode."""
		filler = "C" * 200_000
		with httpx.stream(
			"POST",
			f"{api_base_url}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": f"Filler: {filler}\n\nReply with exactly: PONG"}],
				"stream": True,
			},
			timeout=180.0,
		) as r:
			assert r.status_code == 200
			raw = r.read().decode()

		data_lines = [line[6:] for line in raw.split("\n") if line.startswith("data: ")]
		assert data_lines[-1] == "[DONE]"

		content = ""
		for line in data_lines:
			if line == "[DONE]":
				break
			try:
				chunk = json.loads(line)
				for choice in chunk.get("choices", []):
					delta_content = choice.get("delta", {}).get("content", "")
					if delta_content:
						content += delta_content
			except json.JSONDecodeError:
				pass
		assert "PONG" in content

	def test_large_multi_turn(self, api_base_url):
		"""Multi-turn conversation that exceeds 128KB total."""
		# Build a conversation with multiple large messages.
		messages = []
		for i in range(5):
			messages.append({
				"role": "user",
				"content": f"Part {i}: " + "X" * 30_000,
			})
			messages.append({
				"role": "assistant",
				"content": f"Acknowledged part {i}.",
			})
		messages.append({
			"role": "user",
			"content": "Reply with exactly: PONG",
		})

		r = httpx.post(
			f"{api_base_url}/chat/completions",
			json={"model": "sonnet", "messages": messages},
			timeout=180,
		)
		assert r.status_code == 200
		assert "PONG" in r.json()["choices"][0]["message"]["content"]


# ════════════════════════════════════════════════════════════════
# Text Replacement Rules (uses separate replace-enabled server)
# ════════════════════════════════════════════════════════════════


class TestReplacements:
	def test_response_replacement(self, ccp_replace_server):
		"""Response text should have PONG replaced with REPLACED_OUTPUT."""
		r = httpx.post(
			f"{ccp_replace_server['api_base_url']}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
			},
			timeout=120,
		)
		assert r.status_code == 200
		content = r.json()["choices"][0]["message"]["content"]
		# The word PONG in the response should be replaced.
		assert "REPLACED_OUTPUT" in content

	def test_response_replacement_streaming(self, ccp_replace_server):
		"""Streaming responses should also have replacements applied."""
		with httpx.stream(
			"POST",
			f"{ccp_replace_server['api_base_url']}/chat/completions",
			json={
				"model": "sonnet",
				"messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
				"stream": True,
			},
			timeout=120.0,
		) as r:
			assert r.status_code == 200
			raw = r.read().decode()

		content = ""
		for line in raw.split("\n"):
			if line.startswith("data: ") and line[6:] != "[DONE]":
				try:
					chunk = json.loads(line[6:])
					for choice in chunk.get("choices", []):
						delta_content = choice.get("delta", {}).get("content", "")
						if delta_content:
							content += delta_content
				except json.JSONDecodeError:
					pass

		assert "REPLACED_OUTPUT" in content


# ════════════════════════════════════════════════════════════════
# Conversation Logging
# ════════════════════════════════════════════════════════════════


class TestConversationLog:
	def test_log_file_created(self, ccp_server, client):
		"""After a request, the conversation log file should exist and contain entries."""
		client.chat.completions.create(
			model="sonnet",
			messages=[{"role": "user", "content": "Reply PONG for log test"}],
			stream=False,
		)
		# Give a moment for the log to be written.
		time.sleep(0.5)
		log_file = ccp_server["log_file"]
		assert os.path.isfile(log_file), f"Log file not created at {log_file}"
		content = open(log_file).read()
		assert len(content) > 0
		assert "Inbound OAI body" in content and "OAI response" in content


# ════════════════════════════════════════════════════════════════
# Fingerprint Profiles
# ════════════════════════════════════════════════════════════════


class TestFingerprintProfiles:
	def test_default_profile_outbound_identity(self, ccp_server, client):
		previous_count = _logged_anthropic_request_count(ccp_server["log_file"])
		client.chat.completions.create(
			model="haiku",
			messages=[{"role": "user", "content": "Say OK"}],
			stream=False,
		)
		_assert_profile_identity(ccp_server["log_file"], DEFAULT_PROFILE, previous_count)

	def test_default_profile_streaming_outbound_identity(self, ccp_server, client):
		previous_count = _logged_anthropic_request_count(ccp_server["log_file"])
		stream = client.chat.completions.create(
			model="haiku",
			messages=[{"role": "user", "content": "Say OK"}],
			stream=True,
		)
		for _chunk in stream:
			pass
		_wait_for_logged_profile_request(
			ccp_server["log_file"],
			DEFAULT_PROFILE,
			previous_count,
		)

	def test_user_content_cch_sentinel_is_not_rewritten(self, ccp_server, client):
		previous_count = _logged_anthropic_request_count(ccp_server["log_file"])
		user_text = "Leave this user text alone: cch=00000; 日本語"
		client.chat.completions.create(
			model="haiku",
			messages=[{"role": "user", "content": user_text}],
			stream=False,
		)
		req, raw_body = _wait_for_logged_profile_request(
			ccp_server["log_file"],
			DEFAULT_PROFILE,
			previous_count,
			expected_suffix="*",
		)
		assert user_text in raw_body
		assert "cch=00000;" in raw_body
		assert req["messages"][0]["content"][0]["text"] == user_text
		assert "cc_entrypoint=sdk-cli; cch=00000;" not in req["system"][0]["text"]

	def test_invalid_profile_fails_startup(self, ccp_binary):
		data_dir = tempfile.mkdtemp(prefix="ccp-test-bad-profile-")
		try:
			proc = subprocess.run(
				[
					ccp_binary,
					"--port", "0",
					"--host", "127.0.0.1",
					"--data-dir", data_dir,
					"--fingerprint-profile", "not-a-real-profile",
				],
				capture_output=True,
				text=True,
				timeout=20,
			)
		finally:
			shutil.rmtree(data_dir, ignore_errors=True)

		assert proc.returncode != 0
		stderr = proc.stderr + proc.stdout
		assert "Unknown Claude Code fingerprint profile" in stderr
		assert DEFAULT_PROFILE["name"] in stderr
		assert DEFAULT_PROFILE["version"] in stderr
		assert "2.1.138" not in stderr
