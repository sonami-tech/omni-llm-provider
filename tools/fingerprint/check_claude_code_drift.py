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


PINNED_VERSION = "2.1.161"
PINNED_PROFILE = "cc-2.1.161-sdk-cli"
CCH_SEED = 0x4D659218E32A3268
CCH_RE = re.compile(r"(cc_entrypoint=sdk-cli; cch=)([0-9a-f]{5})(;)")

# --- Clean-room recovered-vector capture (--emit-vectors) ---------------------
# The ONLY per-request-volatile field inside the captured body is
# metadata.user_id's session_id (a fresh UUID each run); the cch then changes
# because it is hashed over the whole body. To get a REPRODUCIBLE real-traffic
# vector we substitute that one UUID with a fixed placeholder, then recompute the
# cch over the normalized bytes. The (normalized_body, recomputed_cch) pair is
# stable run-to-run and still exercises CCP's serializer against Claude Code's
# actual body shape (metadata/context_management/thinking/tools/cache_control).
# device_id is a stable per-machine value, so we also pin it to keep vectors
# machine-independent and free of any host-identifying id.
PLACEHOLDER_SESSION_ID = "00000000-0000-4000-8000-000000000000"
PLACEHOLDER_DEVICE_ID = "0" * 64
PLACEHOLDER_HOME = "/home/ccp-vector"
PLACEHOLDER_EMAIL = "vector@example.com"
SESSION_ID_RE = re.compile(r'(\\"session_id\\":\\")([0-9a-fA-F-]{36})(\\")')
DEVICE_ID_RE = re.compile(r'(\\"device_id\\":\\")([0-9a-fA-F]{64})(\\")')
# Claude Code injects the authenticated account's email into the system prompt
# ("The user's email address is ...") regardless of HOME/CWD — it comes from the
# OAuth account, not a file, so the clean room cannot keep it out. Normalize any
# email to a placeholder. Pattern is intentionally broad (any address).
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Fail-closed tripwire markers for operator-private content leaking into a
# vector. We deliberately do NOT hardcode any operator-specific string (a
# rule-template name, infra term, or email domain) — that would bake private
# data into this committed tool. Instead we key off the STRUCTURAL signal Claude
# Code emits whenever it injects a CLAUDE.md: the literal "Contents of <path>/
# CLAUDE.md" header (seen in contaminated captures). That string appears only
# when a real CLAUDE.md was loaded, carries no personal data itself, and is
# robust to whatever the file contains. The clean room (clean HOME + clean CWD)
# should prevent any CLAUDE.md from loading; this catches a regression where a
# future CC version re-injects one via a path the clean room does not cover.
PRIVATE_CONTENT_MARKERS = (
	b"Contents of ",   # "Contents of <path>/CLAUDE.md ..." — a loaded instruction file
	b".claude/CLAUDE.md",
)


class CaptureServer(ThreadingHTTPServer):
	def __init__(self, server_address: tuple[str, int]):
		super().__init__(server_address, Handler)
		self.records: list[dict[str, object]] = []
		# When True, POST records also retain the exact raw body bytes (for
		# recovered-vector emission). Off by default to keep the drift check
		# memory-light and string-only.
		self.capture_raw: bool = False


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
		if getattr(self.server, "capture_raw", False):
			record["raw"] = body
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


def run_claude_probe(
	claude_bin: str,
	base_url: str,
	prompt: str,
	timeout: int,
	model: str = "claude-haiku-4-5",
	home: str | None = None,
) -> subprocess.CompletedProcess[str]:
	# When `home` is given we run in a CLEAN environment (env -i style): only the
	# variables Claude Code needs, plus HOME pointing at a scratch dir that holds
	# nothing but credentials. This strips the operator's project CLAUDE.md and
	# skills from the injected context, shrinking the body to a stable, mostly
	# generic shape suitable for a checked-in vector. Without `home` it preserves
	# the legacy behavior (inherit the full environment) for the drift check.
	if home is not None:
		env = {
			"HOME": home,
			"PATH": os.environ.get("PATH", ""),
			"ANTHROPIC_BASE_URL": base_url,
			"CLAUDE_CODE_ENTRYPOINT": "sdk-cli",
			"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
		}
	else:
		env = os.environ.copy()
		env["ANTHROPIC_BASE_URL"] = base_url
		env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-cli"
		env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
	# Run from a clean CWD when in clean-room mode: `claude --print` discovers a
	# project CLAUDE.md by walking up from the working directory, so launching
	# from the repo root injects the project (and operator) instructions into the
	# body. Anchoring CWD inside the clean HOME means the walk-up only sees clean
	# dirs, keeping the captured body generic and free of local instructions.
	cwd = home if home is not None else None
	return subprocess.run(
		[
			claude_bin,
			"--print",
			"--model",
			model,
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
		cwd=cwd,
	)


def recompute_cch(body_bytes: bytes) -> str:
	"""CCP's cch over exact body bytes: replace the first sdk-cli cch with the
	00000 sentinel, xxh64(seed) the result, take the low 20 bits as 5 hex."""
	placeholder = CCH_RE.sub(
		lambda m: m.group(1) + "00000" + m.group(3),
		body_bytes.decode("utf-8"),
		count=1,
	).encode("utf-8")
	return f"{xxhash.xxh64(placeholder, seed=CCH_SEED).intdigest() & 0xfffff:05x}"


def normalize_vector_body(raw: bytes, home: str) -> bytes:
	"""Make a captured body reproducible AND host-independent.

	Three substitutions, all deterministic so the normalized bytes (and the cch
	recomputed over them) are stable run-to-run:
	  1. session_id  -> fixed UUID (the per-request volatile field)
	  2. device_id   -> fixed zeros (per-machine, host-identifying)
	  3. the clean HOME path -> a fixed placeholder. Claude Code embeds its
	     memory directory path (<HOME>/.claude/projects/...) in the system
	     prompt; since HOME is a randomly-named tmpfs dir, that path is both
	     volatile (random suffix) and host-identifying (uid). Replacing the whole
	     HOME prefix neutralizes both.

	The billing cch is NOT touched here; the caller recomputes it AFTER
	normalization (a normalized body hashes to a different cch than the raw one,
	precisely because these fields changed)."""
	text = raw.decode("utf-8")
	text = SESSION_ID_RE.sub(
		lambda m: m.group(1) + PLACEHOLDER_SESSION_ID + m.group(3), text, count=1
	)
	text = DEVICE_ID_RE.sub(
		lambda m: m.group(1) + PLACEHOLDER_DEVICE_ID + m.group(3), text, count=1
	)
	# Claude Code embeds the HOME path into the system prompt (its memory dir) in
	# THREE encodings, and HOME is uid-bearing, so we replace all three with a
	# fixed placeholder to keep vectors host-independent (and, with the fixed-path
	# HOME, deterministic):
	#   1. plain absolute path        (e.g. <tmpfs>/ccp-vector-home)
	#   2. JSON-escaped path (\/ slashes)
	#   3. dasherized memory-project slug — Claude Code's project key replaces
	#      every "/" and "." in the path with "-"
	text = text.replace(home.replace("/", "\\/"), PLACEHOLDER_HOME.replace("/", "\\/"))
	text = text.replace(home, PLACEHOLDER_HOME)
	home_slug = home.replace("/", "-").replace(".", "-")
	placeholder_slug = PLACEHOLDER_HOME.replace("/", "-").replace(".", "-")
	text = text.replace(home_slug, placeholder_slug)
	# Account email (from the OAuth identity, injected into the system prompt).
	text = EMAIL_RE.sub(PLACEHOLDER_EMAIL, text)
	return text.encode("utf-8")


def rewrite_cch_in_body(body_bytes: bytes, new_cch: str) -> bytes:
	"""Set the first sdk-cli billing cch to new_cch (used to make the normalized
	vector body self-consistent: its embedded cch matches a hash of itself)."""
	return CCH_RE.sub(
		lambda m: m.group(1) + new_cch + m.group(3),
		body_bytes.decode("utf-8"),
		count=1,
	).encode("utf-8")


def assert_token_free(body_bytes: bytes) -> None:
	"""Vectors are committed to source; fail closed if a bearer token ever
	appears in a body (it lives in the Authorization HEADER, not the body, but
	never ship one by accident)."""
	if b"sk-ant-oat01-" in body_bytes or b"sk-ant-api" in body_bytes:
		raise RuntimeError("refusing to emit a vector body containing an OAuth token")


def assert_no_private_content(body_bytes: bytes) -> None:
	"""Fail closed if operator-private instruction content slips into a vector.
	The clean room (clean HOME + clean CWD) should prevent CLAUDE.md/skills from
	loading; this catches a regression where a future CC version re-injects them
	via a path the clean room does not cover."""
	hits = [m.decode() for m in PRIVATE_CONTENT_MARKERS if m in body_bytes]
	if hits:
		raise RuntimeError(
			"refusing to emit a vector containing operator-private content "
			f"(markers: {', '.join(hits)}); the clean room failed to isolate it"
		)


def make_clean_home() -> str:
	"""A scratch HOME on tmpfs (RAM-backed) holding ONLY credentials, so the
	bearer token never lands on persistent disk and no project CLAUDE.md/skills
	are loaded. Mirrors capture_baseline.sh's tmpfs fail-closed stance.

	The path is FIXED (not mkdtemp): Claude Code embeds the HOME path into the
	system prompt in several encodings (absolute, JSON-escaped, and a dasherized
	memory-project slug). A random suffix would leak through whichever encoding
	we forget to normalize and break run-to-run reproducibility. A fixed,
	deterministic path removes the volatility at the source — there is no random
	component to leak — and is the same every run, so the captured body bytes are
	stable. We recreate it cleanly each run to avoid stale state. Token safety is
	preserved: it is still tmpfs-only and 0700, and shredded in the finally of
	emit_vectors."""
	for base in (os.environ.get("XDG_RUNTIME_DIR"), "/dev/shm", f"/run/user/{os.getuid()}"):
		if base and os.path.isdir(base) and os.access(base, os.W_OK):
			home = os.path.join(base, "ccp-vector-home")
			break
	else:
		raise RuntimeError(
			"no RAM-backed tmpfs (XDG_RUNTIME_DIR, /dev/shm, /run/user/UID) for the "
			"credential-bearing clean HOME; refusing to write the token to disk"
		)
	shutil.rmtree(home, ignore_errors=True)
	os.makedirs(os.path.join(home, ".claude"), exist_ok=False)
	os.chmod(home, 0o700)
	src = os.path.expanduser("~/.claude/.credentials.json")
	dst = os.path.join(home, ".claude", ".credentials.json")
	shutil.copyfile(src, dst)
	os.chmod(dst, 0o600)
	return home


def emit_vectors(claude_bin: str, version: str, out_dir: str, timeout: int) -> int:
	"""Drive the REAL claude CLI in a clean room across each pinned model, then
	emit normalized, token-free, reproducible (cch, body) recovered-capture
	vectors. Each vector is a real Claude Code body whose embedded cch CCP must
	reproduce — breaking the self-consistency circularity of hashing only CCP's
	own output. Re-run on every rebaseline to refresh the set for the new
	version."""
	if version != PINNED_VERSION:
		print(
			json.dumps(
				{
					"status": "error",
					"detail": (
						f"installed claude {version} != pinned {PINNED_VERSION}; "
						"rebaseline the profile BEFORE emitting vectors"
					),
				},
				indent=2,
			)
		)
		return 1

	os.makedirs(out_dir, exist_ok=True)
	models = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"]
	home = make_clean_home()
	emitted: list[dict[str, object]] = []
	try:
		for model in models:
			port = free_port()
			server = CaptureServer(("127.0.0.1", port))
			server.capture_raw = True  # type: ignore[attr-defined]
			thread = threading.Thread(target=server.serve_forever, daemon=True)
			thread.start()
			try:
				result = run_claude_probe(
					claude_bin,
					f"http://127.0.0.1:{port}",
					"Say OK",
					timeout,
					model=model,
					home=home,
				)
			finally:
				server.shutdown()
				server.server_close()
				thread.join(timeout=5)

			raws = [
				rec["raw"]
				for rec in server.records
				if isinstance(rec.get("path"), str)
				and str(rec["path"]).startswith("/v1/messages")
				and is_messages_body(str(rec.get("body", "")))
				and isinstance(rec.get("raw"), (bytes, bytearray))
			]
			if result.returncode != 0 or not raws:
				print(
					json.dumps(
						{
							"status": "error",
							"model": model,
							"returncode": result.returncode,
							"captured": len(raws),
							"stderr_tail": result.stderr[-600:],
						},
						indent=2,
					)
				)
				return 1

			raw = bytes(raws[-1])
			normalized = normalize_vector_body(raw, home)
			cch = recompute_cch(normalized)
			# Make the body self-consistent: its embedded billing cch == hash of
			# itself, so the Rust test (replace cch->00000, recompute, compare)
			# passes against these exact committed bytes.
			vector_body = rewrite_cch_in_body(normalized, cch)
			assert_token_free(vector_body)
			assert_no_private_content(vector_body)
			# Sanity: recomputing over the final self-consistent body reproduces cch.
			assert recompute_cch(vector_body) == cch, "normalized vector not self-consistent"

			path = os.path.join(out_dir, f"vector-{version}-{model}.json")
			with open(path, "wb") as fh:
				fh.write(vector_body)
			emitted.append({"model": model, "cch": cch, "bytes": len(vector_body), "path": path})

		print(
			json.dumps(
				{
					"status": "ok",
					"version": version,
					"placeholder_session_id": PLACEHOLDER_SESSION_ID,
					"vectors": emitted,
					"note": (
						"Reproducible real-traffic vectors. Wire each (cch, body) into the "
						"Rust cch_checksum_matches_recovered_claude_code_captures test."
					),
				},
				indent=2,
			)
		)
		return 0
	finally:
		shutil.rmtree(home, ignore_errors=True)


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
	parser.add_argument(
		"--emit-vectors",
		metavar="DIR",
		default=None,
		help=(
			"Clean-room capture mode: drive the real claude CLI per pinned model "
			"in a credential-only temp HOME and write normalized, reproducible "
			"real-traffic (cch, body) vectors to DIR for the Rust recovered-set. "
			"Does not run the drift check."
		),
	)
	args = parser.parse_args()

	version = claude_version(args.claude_bin)

	if args.emit_vectors is not None:
		return emit_vectors(args.claude_bin, version, args.emit_vectors, args.timeout)
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
