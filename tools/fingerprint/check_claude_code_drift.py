#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "xxhash>=3.5",
# ]
# ///
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
	import xxhash
except ImportError:
	print("missing dependency: install xxhash for Python", file=sys.stderr)
	sys.exit(2)


PINNED_VERSION = "2.1.142"
PINNED_PROFILE = "cc-2.1.142-sdk-cli"
CCH_SEED = 0x4D659218E32A3268
CCH_RE = re.compile(r"(cc_entrypoint=sdk-cli; cch=)([0-9a-f]{5})(;)")


class CaptureServer(ThreadingHTTPServer):
	def __init__(self, server_address: tuple[str, int]):
		super().__init__(server_address, Handler)
		self.records: list[dict[str, object]] = []


class Handler(BaseHTTPRequestHandler):
	def do_GET(self) -> None:
		body = json.dumps({"data": []}).encode()
		self.send_response(200)
		self.send_header("content-type", "application/json")
		self.send_header("content-length", str(len(body)))
		self.end_headers()
		self.wfile.write(body)

	def do_POST(self) -> None:
		length = int(self.headers.get("content-length", "0"))
		body = self.rfile.read(length)
		record = {
			"method": self.command,
			"path": self.path,
			"headers": dict(self.headers.items()),
			"body": body.decode("utf-8", errors="replace"),
		}
		self.server.records.append(record)

		if self.path.startswith("/v1/messages"):
			self.send_response(200)
			self.send_header("content-type", "text/event-stream")
			self.send_header("cache-control", "no-cache")
			self.end_headers()
			for event, data in stream_events():
				self.wfile.write(f"event: {event}\n".encode())
				self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
			self.wfile.flush()
			return

		body = json.dumps({"ok": True}).encode()
		self.send_response(200)
		self.send_header("content-type", "application/json")
		self.send_header("content-length", str(len(body)))
		self.end_headers()
		self.wfile.write(body)

	def log_message(self, _format: str, *_args: object) -> None:
		return


def stream_events() -> list[tuple[str, dict[str, object]]]:
	return [
		(
			"message_start",
			{
				"type": "message_start",
				"message": {
					"id": "msg_drift_check",
					"type": "message",
					"role": "assistant",
					"model": "claude-haiku-4-5-20251001",
					"content": [],
					"stop_reason": None,
					"stop_sequence": None,
					"usage": {"input_tokens": 1, "output_tokens": 0},
				},
			},
		),
		(
			"content_block_start",
			{
				"type": "content_block_start",
				"index": 0,
				"content_block": {"type": "text", "text": ""},
			},
		),
		(
			"content_block_delta",
			{
				"type": "content_block_delta",
				"index": 0,
				"delta": {"type": "text_delta", "text": "OK"},
			},
		),
		("content_block_stop", {"type": "content_block_stop", "index": 0}),
		(
			"message_delta",
			{
				"type": "message_delta",
				"delta": {"stop_reason": "end_turn", "stop_sequence": None},
				"usage": {"output_tokens": 1},
			},
		),
		("message_stop", {"type": "message_stop"}),
	]


def free_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		sock.bind(("127.0.0.1", 0))
		return sock.getsockname()[1]


def claude_version(claude_bin: str) -> str:
	result = subprocess.run(
		[claude_bin, "--version"],
		text=True,
		capture_output=True,
		timeout=20,
	)
	if result.returncode != 0:
		raise RuntimeError(result.stderr.strip() or result.stdout.strip())
	match = re.search(r"(\d+\.\d+\.\d+)", result.stdout + result.stderr)
	return match.group(1) if match else "unknown"


def run_claude_probe(claude_bin: str, base_url: str, prompt: str, timeout: int) -> subprocess.CompletedProcess[str]:
	env = os.environ.copy()
	env["ANTHROPIC_BASE_URL"] = base_url
	env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-cli"
	env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
	return subprocess.run(
		[
			claude_bin,
			"--print",
			"--model",
			"claude-haiku-4-5",
			"--tools",
			"",
			"--no-session-persistence",
			"--",
			prompt,
		],
		text=True,
		capture_output=True,
		timeout=timeout,
		env=env,
	)


def validate_body(body: str) -> tuple[bool, str]:
	match = CCH_RE.search(body)
	if match is None:
		return False, "no sdk-cli cch marker found"
	actual = match.group(2)
	placeholder = CCH_RE.sub(r"\g<1>00000\g<3>", body, count=1).encode()
	expected = f"{xxhash.xxh64(placeholder, seed=CCH_SEED).intdigest() & 0xfffff:05x}"
	if actual != expected:
		return False, f"cch mismatch: actual={actual} expected={expected}"
	if actual == "00000":
		return False, "cch stayed at sentinel 00000"
	return True, f"cch matches pinned algorithm: {actual}"


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Opt-in live drift check for installed Claude Code cch behavior."
	)
	parser.add_argument("--claude-bin", default=shutil.which("claude") or "claude")
	parser.add_argument("--prompt", default="Say OK")
	parser.add_argument("--timeout", type=int, default=60)
	args = parser.parse_args()

	version = claude_version(args.claude_bin)
	port = free_port()
	server = CaptureServer(("127.0.0.1", port))
	thread = threading.Thread(target=server.serve_forever, daemon=True)
	thread.start()

	try:
		result = run_claude_probe(
			args.claude_bin,
			f"http://127.0.0.1:{port}",
			args.prompt,
			args.timeout,
		)
	finally:
		server.shutdown()
		server.server_close()
		thread.join(timeout=5)

	messages = [
		record
		for record in server.records
		if isinstance(record.get("path"), str)
		and str(record["path"]).startswith("/v1/messages")
	]
	candidate_messages = [
		record
		for record in messages
		if is_messages_body(str(record.get("body", "")))
	]
	report = {
		"installed_version": version,
		"pinned_version": PINNED_VERSION,
		"pinned_profile": PINNED_PROFILE,
		"claude_returncode": result.returncode,
		"message_requests": len(messages),
	}

	if version != PINNED_VERSION:
		report["version_status"] = "drift"
	else:
		report["version_status"] = "match"

	if result.returncode != 0:
		report["status"] = "error"
		report["stderr_tail"] = result.stderr[-1200:]
		print(json.dumps(report, indent=2, sort_keys=True))
		return 1

	if not messages:
		report["status"] = "error"
		report["detail"] = "Claude Code completed without a captured /v1/messages request"
		print(json.dumps(report, indent=2, sort_keys=True))
		return 1

	if not candidate_messages:
		report["status"] = "error"
		report["detail"] = "No captured /v1/messages request had both messages and system fields"
		print(json.dumps(report, indent=2, sort_keys=True))
		return 1

	body = str(candidate_messages[-1]["body"])
	ok, detail = validate_body(body)
	report["status"] = "ok" if ok and version == PINNED_VERSION else "drift"
	report["detail"] = detail
	print(json.dumps(report, indent=2, sort_keys=True))
	return 0 if report["status"] == "ok" else 1


def is_messages_body(body: str) -> bool:
	try:
		parsed = json.loads(body)
	except json.JSONDecodeError:
		return False
	return isinstance(parsed, dict) and "messages" in parsed and "system" in parsed


if __name__ == "__main__":
	try:
		sys.exit(main())
	except KeyboardInterrupt:
		sys.exit(130)
	except Exception as exc:
		print(f"error: {exc}", file=sys.stderr)
		sys.exit(1)
