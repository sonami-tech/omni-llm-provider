"""
Pytest fixtures for claude-code-provider integration tests.

Automatically builds the binary and starts a CCP server instance
with isolated config per test session.
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
	"""
	Start a CCP server with no-auth on a random port.
	Yields a dict with connection info. Tears down on session end.
	"""
	port = _free_port()
	data_dir = tempfile.mkdtemp(prefix="ccp-test-")
	log_file = os.path.join(data_dir, "conversations.log")

	proc = subprocess.Popen(
		[
			ccp_binary,
			"--port", str(port),
			"--host", "127.0.0.1",
			"--no-auth",
			"--no-isolate",
			"--max-concurrent", "3",
			"--timeout", "60",
			"--queue-timeout", "10",
			"--max-turns", "2",
			"--log-file", log_file,
			"--data-dir", data_dir,
			"--verbose",
		],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
	)

	base_url = f"http://127.0.0.1:{port}"
	try:
		_wait_for_health(base_url)
	except RuntimeError:
		proc.kill()
		stdout, stderr = proc.communicate(timeout=5)
		pytest.fail(
			f"CCP server failed to start on port {port}.\n"
			f"stdout: {stdout.decode()}\n"
			f"stderr: {stderr.decode()}"
		)

	info = {
		"base_url": base_url,
		"api_base_url": f"{base_url}/v1",
		"port": port,
		"data_dir": data_dir,
		"log_file": log_file,
		"process": proc,
	}

	yield info

	# Teardown.
	proc.terminate()
	try:
		proc.wait(timeout=10)
	except subprocess.TimeoutExpired:
		proc.kill()
		proc.wait(timeout=5)
	shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def ccp_auth_server(ccp_binary):
	"""
	Start a second CCP server WITH auth enabled for auth-specific tests.
	Yields a dict with connection info including the API key.
	"""
	port = _free_port()
	data_dir = tempfile.mkdtemp(prefix="ccp-test-auth-")
	api_key = "sk-test-integration-key-1234"

	proc = subprocess.Popen(
		[
			ccp_binary,
			"--port", str(port),
			"--host", "127.0.0.1",
			"--api-keys", api_key,
			"--no-isolate",
			"--max-concurrent", "2",
			"--timeout", "60",
			"--queue-timeout", "10",
			"--data-dir", data_dir,
		],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
	)

	base_url = f"http://127.0.0.1:{port}"
	try:
		_wait_for_health(base_url)
	except RuntimeError:
		proc.kill()
		stdout, stderr = proc.communicate(timeout=5)
		pytest.fail(
			f"CCP auth server failed to start on port {port}.\n"
			f"stdout: {stdout.decode()}\n"
			f"stderr: {stderr.decode()}"
		)

	info = {
		"base_url": base_url,
		"api_base_url": f"{base_url}/v1",
		"port": port,
		"data_dir": data_dir,
		"api_key": api_key,
		"process": proc,
	}

	yield info

	proc.terminate()
	try:
		proc.wait(timeout=10)
	except subprocess.TimeoutExpired:
		proc.kill()
		proc.wait(timeout=5)
	shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def ccp_replace_server(ccp_binary):
	"""
	Start a CCP server with text replacement rules enabled.
	"""
	port = _free_port()
	data_dir = tempfile.mkdtemp(prefix="ccp-test-replace-")

	# Write a replacement rules TOML file.
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

	proc = subprocess.Popen(
		[
			ccp_binary,
			"--port", str(port),
			"--host", "127.0.0.1",
			"--no-auth",
			"--no-isolate",
			"--max-concurrent", "2",
			"--timeout", "60",
			"--queue-timeout", "10",
			"--data-dir", data_dir,
			"--replace-rules", rules_file,
		],
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
	)

	base_url = f"http://127.0.0.1:{port}"
	try:
		_wait_for_health(base_url)
	except RuntimeError:
		proc.kill()
		stdout, stderr = proc.communicate(timeout=5)
		pytest.fail(
			f"CCP replace server failed to start on port {port}.\n"
			f"stdout: {stdout.decode()}\n"
			f"stderr: {stderr.decode()}"
		)

	info = {
		"base_url": base_url,
		"api_base_url": f"{base_url}/v1",
		"port": port,
		"data_dir": data_dir,
		"process": proc,
	}

	yield info

	proc.terminate()
	try:
		proc.wait(timeout=10)
	except subprocess.TimeoutExpired:
		proc.kill()
		proc.wait(timeout=5)
	shutil.rmtree(data_dir, ignore_errors=True)


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
