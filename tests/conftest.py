"""
Pytest fixtures for claude-code-provider integration tests.

Automatically builds the binary and starts a CCP server instance
with a temporary data directory per test session.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI


def _free_port() -> int:
	"""Find a free TCP port."""
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout: float = 120) -> None:
	"""Block until the server's /health endpoint responds 200."""
	deadline = time.monotonic() + timeout
	while time.monotonic() < deadline:
		try:
			r = httpx.get(f"{base_url}/health", timeout=2)
			if r.status_code == 200:
				return
		except httpx.ConnectError:
			pass
		time.sleep(0.3)
	raise RuntimeError(f"Server at {base_url} did not become healthy within {timeout}s")


def _start_ccp_server(binary: str, name: str, extra_args: list[str], data_dir: str) -> dict:
	"""Spawn a CCP server and wait until it's healthy.

	Redirects stdout/stderr to a file rather than `subprocess.PIPE`. With
	`--verbose` and many sequential requests, the unread pipe buffer fills
	(~64KB) and CCP blocks on its next stderr write — deadlocking all tokio
	tasks and leaving server processes with stuck handlers.

	Returns a dict with base_url, api_base_url, port, data_dir, process.
	On startup failure, calls `pytest.fail` with the tail of the stderr log.
	"""
	port = _free_port()
	stderr_file = os.path.join(data_dir, "ccp.stderr")
	stderr_fp = open(stderr_file, "wb")

	args = [binary, "--port", str(port), "--host", "127.0.0.1", "--data-dir", data_dir]
	args.extend(extra_args)
	proc = subprocess.Popen(args, stdout=stderr_fp, stderr=stderr_fp)

	base_url = f"http://127.0.0.1:{port}"
	try:
		_wait_for_health(base_url)
	except RuntimeError:
		proc.kill()
		proc.wait(timeout=5)
		stderr_fp.close()
		with open(stderr_file, "rb") as f:
			tail = f.read()[-2000:].decode(errors="replace")
		pytest.fail(f"CCP {name} server failed to start on port {port}.\nlast stderr:\n{tail}")

	return {
		"base_url": base_url,
		"api_base_url": f"{base_url}/v1",
		"port": port,
		"data_dir": data_dir,
		"process": proc,
	}


def _stop_ccp_server(info: dict) -> None:
	"""Terminate the server process and remove its data directory."""
	proc = info["process"]
	proc.terminate()
	try:
		proc.wait(timeout=10)
	except subprocess.TimeoutExpired:
		proc.kill()
		proc.wait(timeout=5)
	shutil.rmtree(info["data_dir"], ignore_errors=True)


# ── Session-scoped server ─────────────────────────────────────


@pytest.fixture(scope="session")
def ccp_binary():
	"""Build the CCP binary in release mode and return its path."""
	project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	result = subprocess.run(
		["cargo", "build", "--release"],
		cwd=project_root,
		capture_output=True,
		text=True,
		timeout=300,
	)
	if result.returncode != 0:
		pytest.fail(f"cargo build failed:\n{result.stderr}")
	binary = os.path.join(project_root, "target", "release", "claude-code-provider")
	assert os.path.isfile(binary), f"Binary not found at {binary}"
	return binary


@pytest.fixture(scope="session")
def ccp_server(ccp_binary):
	"""Start a CCP server with no-auth on a random port."""
	data_dir = tempfile.mkdtemp(prefix="ccp-test-")
	log_file = os.path.join(data_dir, "conversations.log")
	info = _start_ccp_server(
		ccp_binary,
		name="default",
		extra_args=[
			"--no-auth",
			"--log-file", log_file,
			"--verbose",
		],
		data_dir=data_dir,
	)
	info["log_file"] = log_file
	yield info
	_stop_ccp_server(info)


@pytest.fixture(scope="session")
def ccp_auth_server(ccp_binary):
	"""Start a second CCP server WITH auth enabled for auth-specific tests."""
	data_dir = tempfile.mkdtemp(prefix="ccp-test-auth-")
	api_key = "sk-test-integration-key-1234"
	info = _start_ccp_server(
		ccp_binary,
		name="auth",
		extra_args=[
			"--api-keys", api_key,
		],
		data_dir=data_dir,
	)
	info["api_key"] = api_key
	yield info
	_stop_ccp_server(info)


@pytest.fixture(scope="session")
def ccp_replace_server(ccp_binary):
	"""Start a CCP server with text replacement rules enabled."""
	data_dir = tempfile.mkdtemp(prefix="ccp-test-replace-")
	rules_file = os.path.join(data_dir, "rules.toml")
	with open(rules_file, "w") as f:
		f.write(
			'[[rule]]\n'
			'scope = "prompt"\n'
			'search = "MAGIC_INPUT"\n'
			'replace = "REPLACED_INPUT"\n'
			'\n'
			'[[rule]]\n'
			'scope = "response"\n'
			'search = "PONG"\n'
			'replace = "REPLACED_OUTPUT"\n'
		)
	info = _start_ccp_server(
		ccp_binary,
		name="replace",
		extra_args=[
			"--no-auth",
			"--replace-rules", rules_file,
		],
		data_dir=data_dir,
	)
	yield info
	_stop_ccp_server(info)


# ── Client fixtures ───────────────────────────────────────────


@pytest.fixture(scope="session")
def client(ccp_server) -> OpenAI:
	"""OpenAI SDK sync client pointed at the test server."""
	return OpenAI(base_url=ccp_server["api_base_url"], api_key="not-needed")


@pytest.fixture(scope="session")
def async_client(ccp_server) -> AsyncOpenAI:
	"""OpenAI SDK async client pointed at the test server."""
	return AsyncOpenAI(base_url=ccp_server["api_base_url"], api_key="not-needed")


@pytest.fixture(scope="session")
def base_url(ccp_server) -> str:
	"""Raw base URL (no /v1 prefix)."""
	return ccp_server["base_url"]


@pytest.fixture(scope="session")
def api_base_url(ccp_server) -> str:
	"""API base URL with /v1 prefix."""
	return ccp_server["api_base_url"]
