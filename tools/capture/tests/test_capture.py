#!/usr/bin/env python3
"""Unit tests for the shared capture framework."""

from __future__ import annotations

import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.capture.approvals import (  # noqa: E402
    ApprovalError,
    LIVE_ENV,
    REFRESH_ENV,
    require_capture_approvals,
)
from tools.capture.cli import main as capture_cli_main  # noqa: E402
from tools.capture.credentials import (  # noqa: E402
    CredentialError,
    StagedCredentials,
    codex_base_url_host,
    codex_primary_api_host,
    force_claude_expiry_stale,
    force_codex_expiry_stale,
    force_grok_expiry_stale,
    plan_credentials,
    stage_credentials,
)
from tools.capture.extract import (  # noqa: E402
    ExtractionError,
    extract_jsonl_markdown,
    hosts_in_jsonl,
    validate_expected_hosts,
)
from tools.capture.providers import (  # noqa: E402
    build_provider_commands,
    build_provider_env,
    command_plan,
    provider_command_stdin,
    required_hosts_for_validation,
)
from tools.capture.redaction import redact_body, redact_header, redact_text  # noqa: E402
from tools.capture.runner import (  # noqa: E402
    CaptureError,
    _capture_signal_cleanup,
    _install_capture_signal_handlers,
    _restore_capture_signal_handlers,
    _wait_for_flow_flush,
    run_capture,
)
from tools.capture.workdir import CaptureWorkdir  # noqa: E402


class ApprovalTests(unittest.TestCase):
    def test_general_capture_requires_live_approval(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ApprovalError):
                require_capture_approvals(mode="general", dry_run=False)

    def test_refresh_requires_both_approvals(self) -> None:
        with mock.patch.dict(os.environ, {LIVE_ENV: "1"}, clear=True):
            with self.assertRaises(ApprovalError):
                require_capture_approvals(mode="refresh", dry_run=False)

    def test_refresh_ok_with_both_flags(self) -> None:
        flags = require_capture_approvals(
            mode="refresh",
            dry_run=False,
            live_flag=True,
            refresh_flag=True,
        )
        self.assertTrue(flags.live_capture)
        self.assertTrue(flags.refresh_capture)

    def test_dry_run_skips_approval(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            flags = require_capture_approvals(mode="refresh", dry_run=True)
        self.assertFalse(flags.live_capture)


class CredentialStagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fake_home = Path(self.temp.name) / "real-home"
        self.fake_home.mkdir()
        self.clean_home = Path(self.temp.name) / "clean-home"
        self.clean_home.mkdir()
        self.clean_codex = Path(self.temp.name) / "clean-codex"
        self.clean_codex.mkdir()

    def _write_claude(self, home: Path) -> None:
        creds = {
            "claudeAiOauth": {
                "accessToken": "token-claude",
                "expiresAt": 4_102_444_800_000,
            }
        }
        path = home / ".claude"
        path.mkdir(parents=True)
        (path / ".credentials.json").write_text(json.dumps(creds), encoding="utf-8")

    def _write_grok_oidc(self, home: Path) -> None:
        creds = {
            "https://auth.x.ai::client": {
                "key": "jwt-grok",
                "auth_mode": "oidc",
                "expires_at": "2999-01-01T00:00:00Z",
            }
        }
        path = home / ".grok"
        path.mkdir(parents=True)
        (path / "auth.json").write_text(json.dumps(creds), encoding="utf-8")

    def _write_codex(self, home: Path) -> None:
        (home / "config.toml").write_text(
            'model = "gpt-5"\n'
            'model_provider = "proxy"\n'
            '[model_providers.proxy]\n'
            'base_url = "https://proxy.example.com/v1"\n',
            encoding="utf-8",
        )
        auth = {
            "tokens": {
                "access_token": "codex-token",
                "expires_at": "2999-01-01T00:00:00Z",
            }
        }
        (home / "auth.json").write_text(json.dumps(auth), encoding="utf-8")

    def test_stage_claude_copies_only_needed_files(self) -> None:
        self._write_claude(self.fake_home)
        staged = stage_credentials(
            provider="claude",
            clean_home=self.clean_home,
            clean_codex_home=None,
            mode="general",
            source_home=self.fake_home,
        )
        dest = self.clean_home / ".claude" / ".credentials.json"
        self.assertTrue(dest.is_file())
        self.assertEqual(staged.env_overrides["HOME"], str(self.clean_home))
        self.assertEqual(staged.env_overrides["CLAUDE_CREDENTIALS_PATH"], str(dest))

    def test_refresh_claude_forces_expiry_stale(self) -> None:
        self._write_claude(self.fake_home)
        staged = stage_credentials(
            provider="claude",
            clean_home=self.clean_home,
            clean_codex_home=None,
            mode="refresh",
            source_home=self.fake_home,
        )
        dest = staged.copied_paths[0]
        data = json.loads(dest.read_text(encoding="utf-8"))
        self.assertEqual(data["claudeAiOauth"]["expiresAt"], 1_000)
        original = json.loads(
            (self.fake_home / ".claude" / ".credentials.json").read_text(encoding="utf-8")
        )
        self.assertEqual(original["claudeAiOauth"]["expiresAt"], 4_102_444_800_000)

    def test_refresh_grok_forces_oidc_expiry_stale(self) -> None:
        self._write_grok_oidc(self.fake_home)
        staged = stage_credentials(
            provider="grok",
            clean_home=self.clean_home,
            clean_codex_home=None,
            mode="refresh",
            source_home=self.fake_home,
        )
        dest = self.clean_home / ".grok" / "auth.json"
        data = json.loads(dest.read_text(encoding="utf-8"))
        entry = next(iter(data.values()))
        self.assertEqual(entry["expires_at"], "2000-01-01T00:00:00.000000Z")

    def test_refresh_grok_fails_on_static_api_key_only(self) -> None:
        path = self.fake_home / ".xai"
        path.mkdir(parents=True)
        (path / ".credentials.json").write_text(
            json.dumps({"apiKey": "static-key", "xaiApiKey": "static-key"}),
            encoding="utf-8",
        )
        with self.assertRaises(CredentialError) as ctx:
            stage_credentials(
                provider="grok",
                clean_home=self.clean_home,
                clean_codex_home=None,
                mode="refresh",
                source_home=self.fake_home,
            )
        self.assertIn("OIDC auth.json", str(ctx.exception))

    def test_stage_grok_custom_xai_credentials_path(self) -> None:
        custom = self.fake_home / "my-xai-creds.json"
        custom.write_text(
            json.dumps({"apiKey": "static-key", "xaiApiKey": "static-key"}),
            encoding="utf-8",
        )
        with mock.patch.dict(
            os.environ, {"XAI_CREDENTIALS_PATH": str(custom)}, clear=False
        ):
            staged = stage_credentials(
                provider="grok",
                clean_home=self.clean_home,
                clean_codex_home=None,
                mode="general",
            )
        dest = self.clean_home / "my-xai-creds.json"
        self.assertTrue(dest.is_file())
        self.assertEqual(staged.env_overrides["XAI_CREDENTIALS_PATH"], str(dest))

    def test_refresh_grok_ignores_static_key_when_oidc_exists(self) -> None:
        path = self.fake_home / ".xai"
        path.mkdir(parents=True)
        (path / ".credentials.json").write_text(
            json.dumps({"apiKey": "static-key"}),
            encoding="utf-8",
        )
        self._write_grok_oidc(self.fake_home)
        staged = stage_credentials(
            provider="grok",
            clean_home=self.clean_home,
            clean_codex_home=None,
            mode="refresh",
            source_home=self.fake_home,
        )
        self.assertTrue((self.clean_home / ".grok" / "auth.json").is_file())
        self.assertFalse((self.clean_home / ".xai" / ".credentials.json").exists())
        self.assertNotIn("XAI_CREDENTIALS_PATH", staged.env_overrides)

    def test_refresh_codex_forces_expiry_stale(self) -> None:
        self._write_codex(self.fake_home)
        staged = stage_credentials(
            provider="codex",
            clean_home=self.clean_home,
            clean_codex_home=self.clean_codex,
            mode="refresh",
            source_codex_home=self.fake_home,
        )
        dest = self.clean_codex / "auth.json"
        data = json.loads(dest.read_text(encoding="utf-8"))
        self.assertEqual(data["tokens"]["expires_at"], "2000-01-01T00:00:00.000000Z")

    def test_refresh_codex_preserves_refresh_token_expiry(self) -> None:
        auth = {
            "tokens": {
                "access_token": "codex-token",
                "expires_at": "2999-01-01T00:00:00Z",
                "expires_in": 3600,
                "refresh_token_expires_at": "2999-06-01T00:00:00Z",
            },
            "last_refresh": "2999-01-01T00:00:00Z",
        }
        (self.fake_home / "auth.json").write_text(json.dumps(auth), encoding="utf-8")
        stage_credentials(
            provider="codex",
            clean_home=self.clean_home,
            clean_codex_home=self.clean_codex,
            mode="refresh",
            source_codex_home=self.fake_home,
        )
        data = json.loads((self.clean_codex / "auth.json").read_text(encoding="utf-8"))
        self.assertEqual(
            data["tokens"]["refresh_token_expires_at"],
            "2999-06-01T00:00:00Z",
        )
        self.assertEqual(data["tokens"]["expires_at"], "2000-01-01T00:00:00.000000Z")
        self.assertEqual(data["tokens"]["expires_in"], 0)
        self.assertEqual(data["last_refresh"], "2000-01-01T00:00:00.000000Z")

    def test_stage_codex_uses_clean_codex_home(self) -> None:
        self._write_codex(self.fake_home)
        staged = stage_credentials(
            provider="codex",
            clean_home=self.clean_home,
            clean_codex_home=self.clean_codex,
            mode="general",
            source_codex_home=self.fake_home,
        )
        self.assertEqual(staged.env_overrides["CODEX_HOME"], str(self.clean_codex))
        self.assertTrue((self.clean_codex / "config.toml").is_file())
        self.assertTrue((self.clean_codex / "auth.json").is_file())

    def test_refresh_codex_requires_auth_json(self) -> None:
        (self.fake_home / "config.toml").write_text('model = "gpt-5"\n', encoding="utf-8")
        with self.assertRaises(CredentialError) as ctx:
            stage_credentials(
                provider="codex",
                clean_home=self.clean_home,
                clean_codex_home=self.clean_codex,
                mode="refresh",
                source_codex_home=self.fake_home,
            )
        self.assertIn("auth.json", str(ctx.exception))


class RedactionTests(unittest.TestCase):
    def test_header_redaction(self) -> None:
        self.assertEqual(redact_header("Authorization", "Bearer secret"), "<redacted>")
        self.assertEqual(redact_header("User-Agent", "claude"), "claude")

    def test_header_redaction_additional_sensitive_headers(self) -> None:
        for name in (
            "Set-Cookie",
            "set-cookie",
            "Proxy-Authorization",
            "X-Auth-Token",
            "X-Access-Token",
        ):
            with self.subTest(header=name):
                self.assertEqual(redact_header(name, "secret-value"), "<redacted>")

    def test_text_redaction_token_pairs(self) -> None:
        redacted = redact_text("access_token=abc123&refreshToken:rt456")
        self.assertNotIn("abc123", redacted)
        self.assertNotIn("rt456", redacted)
        self.assertIn("<redacted>", redacted)

    def test_body_redaction(self) -> None:
        body = {"api_key": "secret", "model": "claude-sonnet"}
        redacted = redact_body(body)
        self.assertEqual(redacted["api_key"], "<redacted>")
        self.assertEqual(redacted["model"], "claude-sonnet")

    def test_body_redaction_oauth_tokens(self) -> None:
        body = {
            "access_token": "at",
            "refresh_token": "rt",
            "id_token": "id",
            "bearer_token": "bt",
            "client_secret": "cs",
            "secret": "s",
            "session_token": "st",
            "jwt": "j",
            "apiKey": "ak",
            "xaiApiKey": "xk",
            "OPENAI_API_KEY": "oa",
            "CODEX_ACCESS_TOKEN": "ca",
            "nested": {"refresh_token": "nested-rt"},
            "model": "grok",
        }
        redacted = redact_body(body)
        for key in (
            "access_token",
            "refresh_token",
            "id_token",
            "bearer_token",
            "client_secret",
            "secret",
            "session_token",
            "jwt",
            "apiKey",
            "xaiApiKey",
            "OPENAI_API_KEY",
            "CODEX_ACCESS_TOKEN",
        ):
            self.assertEqual(redacted[key], "<redacted>", key)
        self.assertEqual(redacted["nested"]["refresh_token"], "<redacted>")
        self.assertEqual(redacted["model"], "grok")

    def test_body_redaction_camelcase_oauth_tokens(self) -> None:
        body = {
            "accessToken": "at",
            "refreshToken": "rt",
            "idToken": "id",
            "clientSecret": "cs",
            "sessionToken": "st",
            "model": "claude",
        }
        redacted = redact_body(body)
        for key in (
            "accessToken",
            "refreshToken",
            "idToken",
            "clientSecret",
            "sessionToken",
        ):
            self.assertEqual(redacted[key], "<redacted>", key)
        self.assertEqual(redacted["model"], "claude")

    def test_body_redaction_string_payload(self) -> None:
        self.assertEqual(redact_body("access_token=abc123"), "access_token=<redacted>")


class ExtractionTests(unittest.TestCase):
    def test_jsonl_extraction_redacts_query_secrets(self) -> None:
        record = {
            "method": "GET",
            "url": (
                "https://auth.x.ai/callback?"
                "access_token=secret-at&refresh_token=secret-rt&code=oauth-code&state=ok"
            ),
            "host": "auth.x.ai",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(json.dumps(record) + "\n")
            path = Path(handle.name)
        try:
            buf = io.StringIO()
            extract_jsonl_markdown(path, provider="grok", out=buf)
            out = buf.getvalue()
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("<redacted>", out)
        self.assertIn("state=ok", out)
        self.assertNotIn("secret-at", out)
        self.assertNotIn("secret-rt", out)
        self.assertNotIn("oauth-code", out)

    def test_jsonl_extraction_redacts_auth(self) -> None:
        record = {
            "method": "POST",
            "url": "https://api.x.ai/v1/chat/completions",
            "headers": {"Authorization": "Bearer secret", "User-Agent": "grok"},
            "body": json.dumps({"model": "grok", "api_key": "secret"}),
            "status": 200,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(json.dumps(record) + "\n")
            path = Path(handle.name)
        try:
            buf = io.StringIO()
            extract_jsonl_markdown(path, out=buf)
            out = buf.getvalue()
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("<redacted>", out)
        self.assertIn("grok", out)
        self.assertNotIn("Bearer secret", out)

    def test_validate_hosts_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_expected_hosts([], ["api.x.ai"])
        with self.assertRaises(RuntimeError):
            validate_expected_hosts(["example.com"], ["api.x.ai"])

    def test_validate_hosts_requires_primary_not_auth_only(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            validate_expected_hosts(
                ["auth.x.ai"],
                ["cli-chat-proxy.grok.com"],
            )
        self.assertIn("cli-chat-proxy.grok.com", str(ctx.exception))

    def test_validate_hosts_grok_refresh_requires_both_api_and_auth(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_expected_hosts(
                ["cli-chat-proxy.grok.com"],
                ["cli-chat-proxy.grok.com", "auth.x.ai"],
            )
        with self.assertRaises(RuntimeError):
            validate_expected_hosts(
                ["auth.x.ai"],
                ["cli-chat-proxy.grok.com", "auth.x.ai"],
            )
        validate_expected_hosts(
            ["cli-chat-proxy.grok.com", "auth.x.ai"],
            ["cli-chat-proxy.grok.com", "auth.x.ai"],
        )

    def test_validate_hosts_claude_refresh_requires_api_anthropic(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_expected_hosts(["claude.ai"], ["api.anthropic.com"])
        validate_expected_hosts(["api.anthropic.com"], ["api.anthropic.com"])

    def test_codex_host_detection_uses_model_provider_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                'model_provider = "proxy"\n'
                '[model_providers.proxy]\n'
                'base_url = "https://proxy.example.com/v1"\n',
                encoding="utf-8",
            )
            self.assertEqual(codex_primary_api_host(config), "proxy.example.com")

            config.write_text(
                'model_provider = "OpenAI"\n'
                '[model_providers.OpenAI]\n'
                'base_url = "https://custom.openai.example/v1"\n',
                encoding="utf-8",
            )
            self.assertEqual(codex_base_url_host(config), "custom.openai.example")

            config.write_text('openai_base_url = "https://alt.example.com/v1"\n', encoding="utf-8")
            self.assertEqual(codex_primary_api_host(config), "alt.example.com")

    def test_required_hosts_codex_uses_custom_base_url(self) -> None:
        staged = plan_credentials(provider="codex", mode="general")
        required = required_hosts_for_validation("codex", "general", staged)
        self.assertEqual(required, ("api.openai.com",))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_codex = root / "clean-codex"
            clean_codex.mkdir()
            (clean_codex / "config.toml").write_text(
                'model_provider = "proxy"\n'
                '[model_providers.proxy]\n'
                'base_url = "https://proxy.example.com/v1"\n',
                encoding="utf-8",
            )
            staged_custom = StagedCredentials(
                provider="codex",
                clean_home=root / "clean-home",
                clean_codex_home=clean_codex,
                copied_paths=(),
                source_paths=(),
                env_overrides={},
            )
            required_custom = required_hosts_for_validation("codex", "general", staged_custom)
            self.assertEqual(required_custom, ("proxy.example.com",))
            with self.assertRaises(RuntimeError):
                validate_expected_hosts(["api.openai.com"], required_custom)

    def test_hosts_in_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(
                json.dumps({"url": "https://cli-chat-proxy.grok.com/v1/chat/completions"})
                + "\n"
            )
            path = Path(handle.name)
        try:
            hosts = hosts_in_jsonl(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("cli-chat-proxy.grok.com", hosts)

    def test_jsonl_extraction_filters_expected_hosts(self) -> None:
        records = [
            {
                "method": "POST",
                "url": "https://auth.x.ai/oauth/token",
                "headers": {"Authorization": "Bearer secret"},
                "body": {"refresh_token": "rt", "client_secret": "cs"},
                "status": 200,
                "response_body": {"access_token": "at"},
            },
            {
                "method": "POST",
                "url": "https://example.com/ignore",
                "body": {"ignored": True},
            },
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
            path = Path(handle.name)
        try:
            buf = io.StringIO()
            extract_jsonl_markdown(
                path,
                provider="grok",
                out=buf,
                require_matches=True,
            )
            out = buf.getvalue()
        finally:
            path.unlink(missing_ok=True)
        self.assertIn("auth.x.ai", out)
        self.assertNotIn("example.com", out)
        self.assertIn("<redacted>", out)

    def test_jsonl_extraction_codex_custom_base_url(self) -> None:
        record = {
            "method": "POST",
            "url": "https://proxy.example.com/v1/responses",
            "headers": {"Authorization": "Bearer secret"},
            "body": {"model": "gpt-5", "input": "hi"},
            "status": 200,
            "response_body": {"output": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                'model_provider = "proxy"\n'
                '[model_providers.proxy]\n'
                'base_url = "https://proxy.example.com/v1"\n',
                encoding="utf-8",
            )
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
                handle.write(json.dumps(record) + "\n")
                path = Path(handle.name)
            try:
                buf = io.StringIO()
                extract_jsonl_markdown(
                    path,
                    provider="codex",
                    codex_config=config,
                    out=buf,
                    require_matches=True,
                )
                out = buf.getvalue()
            finally:
                path.unlink(missing_ok=True)
        self.assertIn("proxy.example.com", out)
        self.assertIn("top-level keys", out)

    def test_jsonl_extraction_fails_on_zero_matches(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(
                json.dumps({"url": "https://example.com/ignore", "body": {}}) + "\n"
            )
            path = Path(handle.name)
        try:
            with self.assertRaises(ExtractionError):
                extract_jsonl_markdown(
                    path,
                    filter_hosts=["api.x.ai"],
                    require_matches=True,
                    out=io.StringIO(),
                )
        finally:
            path.unlink(missing_ok=True)

    def test_jsonl_extraction_skips_hostless_records_when_filter_active(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(json.dumps({"path": "/v1/messages", "body": {"model": "claude"}}) + "\n")
            path = Path(handle.name)
        try:
            with self.assertRaises(ExtractionError):
                extract_jsonl_markdown(
                    path,
                    filter_hosts=["api.anthropic.com"],
                    require_matches=True,
                    out=io.StringIO(),
                )
            buf = io.StringIO()
            extract_jsonl_markdown(
                path,
                filter_hosts=["api.anthropic.com"],
                require_matches=False,
                out=buf,
            )
            self.assertEqual(buf.getvalue(), "")
        finally:
            path.unlink(missing_ok=True)

    def test_extract_cli_fails_closed_on_zero_matches(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
            handle.write(
                json.dumps({"url": "https://example.com/ignore", "body": {}}) + "\n"
            )
            path = Path(handle.name)
        try:
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                rc = capture_cli_main(
                    ["extract", "jsonl", str(path), "--filter-host", "api.x.ai"]
                )
            self.assertEqual(rc, 1)
            with mock.patch("sys.stderr", new_callable=io.StringIO):
                rc = capture_cli_main(
                    [
                        "extract",
                        "jsonl",
                        str(path),
                        "--filter-host",
                        "api.x.ai",
                        "--allow-empty",
                    ]
                )
            self.assertEqual(rc, 0)
        finally:
            path.unlink(missing_ok=True)

    def test_extract_cli_codex_config_derives_filter_hosts(self) -> None:
        record = {
            "method": "POST",
            "url": "https://proxy.example.com/v1/responses",
            "body": {"model": "gpt-5", "input": "hi"},
            "status": 200,
            "response_body": {"output": []},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                'model_provider = "proxy"\n'
                '[model_providers.proxy]\n'
                'base_url = "https://proxy.example.com/v1"\n',
                encoding="utf-8",
            )
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
                handle.write(json.dumps(record) + "\n")
                path = Path(handle.name)
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = str(_REPO)
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                env["PYTHONWARNINGS"] = "error"
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "tools.capture",
                        "extract",
                        "jsonl",
                        str(path),
                        "--provider",
                        "codex",
                        "--codex-config",
                        str(config),
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=False,
                )
            finally:
                path.unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("proxy.example.com", result.stdout)

    def test_extract_cli_codex_filter_host_overrides_config(self) -> None:
        record = {
            "method": "POST",
            "url": "https://proxy.example.com/v1/responses",
            "body": {"model": "gpt-5", "input": "hi"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                'model_provider = "proxy"\n'
                '[model_providers.proxy]\n'
                'base_url = "https://proxy.example.com/v1"\n',
                encoding="utf-8",
            )
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as handle:
                handle.write(json.dumps(record) + "\n")
                path = Path(handle.name)
            try:
                with mock.patch("sys.stderr", new_callable=io.StringIO):
                    rc = capture_cli_main(
                        [
                            "extract",
                            "jsonl",
                            str(path),
                            "--provider",
                            "codex",
                            "--codex-config",
                            str(config),
                            "--filter-host",
                            "api.x.ai",
                        ]
                    )
                self.assertEqual(rc, 1)
                with mock.patch("sys.stderr", new_callable=io.StringIO):
                    rc = capture_cli_main(
                        [
                            "extract",
                            "jsonl",
                            str(path),
                            "--provider",
                            "codex",
                            "--codex-config",
                            str(config),
                            "--filter-host",
                            "api.x.ai",
                            "--allow-empty",
                        ]
                    )
                self.assertEqual(rc, 0)
            finally:
                path.unlink(missing_ok=True)


class ProviderCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source_home = Path(self.temp.name) / "source-home"
        self.source_home.mkdir()
        self.source_codex = Path(self.temp.name) / "source-codex"
        self.source_codex.mkdir()
        self.clean_home = Path(self.temp.name) / "clean-home"
        self.clean_home.mkdir()
        self.clean_codex = Path(self.temp.name) / "clean-codex"
        self.clean_codex.mkdir()
        (self.source_home / ".claude").mkdir()
        creds = self.source_home / ".claude" / ".credentials.json"
        creds.write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "x", "expiresAt": 9}}),
            encoding="utf-8",
        )

    def test_claude_command_plan_includes_default_and_models(self) -> None:
        staged = stage_credentials(
            provider="claude",
            clean_home=self.clean_home,
            clean_codex_home=None,
            mode="general",
            source_home=self.source_home,
        )
        commands = build_provider_commands(
            provider="claude",
            staged=staged,
            port=12345,
            models=["claude-haiku-4-5"],
        )
        self.assertEqual(len(commands), 2)
        self.assertNotIn("--model", commands[0])
        self.assertIn("--model", commands[1])
        self.assertEqual(commands[1][commands[1].index("--model") + 1], "claude-haiku-4-5")

    def test_grok_command_plan_includes_default_and_models(self) -> None:
        (self.source_home / ".grok").mkdir(exist_ok=True)
        (self.source_home / ".grok" / "auth.json").write_text(
            '{"https://auth.x.ai::x":{"key":"jwt","auth_mode":"oidc","expires_at":"2999-01-01T00:00:00Z"}}',
            encoding="utf-8",
        )
        staged = stage_credentials(
            provider="grok",
            clean_home=self.clean_home,
            clean_codex_home=None,
            mode="general",
            source_home=self.source_home,
        )
        commands = build_provider_commands(
            provider="grok",
            staged=staged,
            port=12345,
            models=["grok-4.5", "grok-4-3"],
        )
        self.assertEqual(len(commands), 3)
        self.assertNotIn("--model", commands[0])
        self.assertEqual(commands[0][commands[0].index("--single") + 1], "Say OK")
        self.assertEqual(commands[1][commands[1].index("--model") + 1], "grok-4.5")
        self.assertEqual(commands[2][commands[2].index("--model") + 1], "grok-4-3")

    def test_all_provider_dry_run_plans(self) -> None:
        for provider in ("claude", "grok", "codex"):
            with self.subTest(provider=provider):
                if provider == "codex":
                    (self.source_codex / "config.toml").write_text(
                        'model = "gpt-5"\n', encoding="utf-8"
                    )
                    (self.source_codex / "auth.json").write_text(
                        '{"tokens":{"access_token":"t","expires_at":"2999-01-01T00:00:00Z"}}',
                        encoding="utf-8",
                    )
                    staged = stage_credentials(
                        provider="codex",
                        clean_home=self.clean_home,
                        clean_codex_home=self.clean_codex,
                        mode="general",
                        source_codex_home=self.source_codex,
                    )
                elif provider == "grok":
                    (self.source_home / ".grok").mkdir(exist_ok=True)
                    (self.source_home / ".grok" / "auth.json").write_text(
                        '{"https://auth.x.ai::x":{"key":"jwt","auth_mode":"oidc","expires_at":"2999-01-01T00:00:00Z"}}',
                        encoding="utf-8",
                    )
                    staged = stage_credentials(
                        provider="grok",
                        clean_home=self.clean_home,
                        clean_codex_home=None,
                        mode="general",
                        source_home=self.source_home,
                    )
                else:
                    staged = stage_credentials(
                        provider="claude",
                        clean_home=self.clean_home,
                        clean_codex_home=None,
                        mode="general",
                        source_home=self.source_home,
                    )
                plan = command_plan(provider=provider, mode="general", staged=staged, port=9999)
                self.assertEqual(plan["provider"], provider)
                self.assertIn("commands", plan)
                self.assertIn("expected_hosts", plan)

    def test_grok_env_uses_proxy(self) -> None:
        (self.source_home / ".grok").mkdir(exist_ok=True)
        (self.source_home / ".grok" / "auth.json").write_text(
            '{"https://auth.x.ai::x":{"key":"jwt","auth_mode":"oidc","expires_at":"2999-01-01T00:00:00Z"}}',
            encoding="utf-8",
        )
        staged = stage_credentials(
            provider="grok",
            clean_home=self.clean_home,
            clean_codex_home=None,
            mode="general",
            source_home=self.source_home,
        )
        env = build_provider_env(
            provider="grok",
            staged=staged,
            port=8080,
            path_value="/usr/bin",
        )
        self.assertEqual(env["HTTP_PROXY"], "http://127.0.0.1:8080")
        self.assertEqual(env["HOME"], str(self.clean_home))

    def test_claude_env_uses_reverse_proxy(self) -> None:
        staged = stage_credentials(
            provider="claude",
            clean_home=self.clean_home,
            clean_codex_home=None,
            mode="general",
            source_home=self.source_home,
        )
        env = build_provider_env(
            provider="claude",
            staged=staged,
            port=8080,
            path_value="/usr/bin",
        )
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8080")
        self.assertNotIn("HTTP_PROXY", env)

    def test_claude_env_scrubs_anthropic_api_keys(self) -> None:
        staged = stage_credentials(
            provider="claude",
            clean_home=self.clean_home,
            clean_codex_home=None,
            mode="general",
            source_home=self.source_home,
        )
        with mock.patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "ambient-key",
                "ANTHROPIC_AUTH_TOKEN": "ambient-token",
            },
            clear=False,
        ):
            env = build_provider_env(
                provider="claude",
                staged=staged,
                port=8080,
                path_value="/usr/bin",
            )
        self.assertNotIn("ANTHROPIC_API_KEY", env)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", env)

    def test_grok_command_uses_single_flag(self) -> None:
        (self.source_home / ".grok").mkdir(exist_ok=True)
        (self.source_home / ".grok" / "auth.json").write_text(
            '{"https://auth.x.ai::x":{"key":"jwt","auth_mode":"oidc","expires_at":"2999-01-01T00:00:00Z"}}',
            encoding="utf-8",
        )
        staged = stage_credentials(
            provider="grok",
            clean_home=self.clean_home,
            clean_codex_home=None,
            mode="general",
            source_home=self.source_home,
        )
        commands = build_provider_commands(
            provider="grok",
            staged=staged,
            port=12345,
            prompt="Say OK",
        )
        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertEqual(cmd[0], "grok")
        self.assertIn("--single", cmd)
        self.assertEqual(cmd[cmd.index("--single") + 1], "Say OK")
        self.assertIn("--output-format", cmd)
        self.assertIn("json", cmd)
        self.assertNotIn("-p", cmd)

    def test_codex_command_disables_mcp_and_reads_stdin(self) -> None:
        (self.source_codex / "config.toml").write_text('model = "gpt-5"\n', encoding="utf-8")
        (self.source_codex / "auth.json").write_text(
            '{"tokens":{"access_token":"t","expires_at":"2999-01-01T00:00:00Z"}}',
            encoding="utf-8",
        )
        staged = stage_credentials(
            provider="codex",
            clean_home=self.clean_home,
            clean_codex_home=self.clean_codex,
            mode="general",
            source_codex_home=self.source_codex,
        )
        commands = build_provider_commands(
            provider="codex",
            staged=staged,
            port=12345,
            prompt="Say OK",
        )
        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertEqual(cmd[:3], ["codex", "exec", "-c"])
        self.assertEqual(cmd[3], "mcp_servers={}")
        self.assertIn("--skip-git-repo-check", cmd)
        self.assertIn("--ephemeral", cmd)
        self.assertEqual(cmd[-1], "-")
        self.assertTrue(provider_command_stdin(cmd))
        self.assertNotIn("--no-mcp", cmd)


class WorkdirTests(unittest.TestCase):
    def test_workdir_permissions_and_cleanup(self) -> None:
        with mock.patch(
            "tools.capture.workdir.find_tmpfs_base",
            return_value=Path(tempfile.gettempdir()),
        ):
            work = CaptureWorkdir.create(
                provider="claude",
                keep_flow=False,
                allow_tmpfs_fallback=True,
            )
            try:
                mode = work.root.stat().st_mode
                self.assertEqual(mode & stat.S_IRWXG, 0)
                self.assertEqual(mode & stat.S_IRWXO, 0)
                work.flow_path.write_text("secret", encoding="utf-8")
                creds = work.clean_home / ".claude" / ".credentials.json"
                creds.parent.mkdir(parents=True)
                creds.write_text('{"token":"secret"}', encoding="utf-8")
                work.remove_all()
                self.assertFalse(work.root.exists())
            finally:
                if work.root.exists():
                    work.remove_all()

    def test_keep_flow_warns_and_retains_workdir(self) -> None:
        with mock.patch(
            "tools.capture.workdir.find_tmpfs_base",
            return_value=Path(tempfile.gettempdir()),
        ):
            work = CaptureWorkdir.create(
                provider="grok",
                keep_flow=True,
                allow_tmpfs_fallback=True,
            )
            try:
                work.flow_path.write_text("secret", encoding="utf-8")
                creds = work.clean_home / ".grok" / "auth.json"
                creds.parent.mkdir(parents=True)
                creds.write_text('{"token":"secret"}', encoding="utf-8")
                with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                    work.remove_all()
                    warning = err.getvalue()
                self.assertTrue(work.root.exists())
                self.assertTrue(work.flow_path.exists())
                self.assertTrue(creds.exists())
                self.assertIn("KEEP_FLOW", warning)
                self.assertIn("staged credential copies", warning)
            finally:
                shutil_rmtree = __import__("shutil").rmtree
                shutil_rmtree(work.root, ignore_errors=True)

    def test_keep_flow_fallback_warns_persistent_disk_not_tmpfs(self) -> None:
        with mock.patch(
            "tools.capture.workdir.find_tmpfs_base",
            return_value=Path(tempfile.gettempdir()),
        ):
            with mock.patch("tools.capture.workdir.is_ramfs", return_value=False):
                work = CaptureWorkdir.create(
                    provider="grok",
                    keep_flow=True,
                    allow_tmpfs_fallback=True,
                )
                try:
                    work.flow_path.write_text("secret", encoding="utf-8")
                    with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
                        work.remove_all()
                        warning = err.getvalue()
                    self.assertIn("persistent disk", warning)
                    self.assertNotIn("tmpfs-only", warning)
                finally:
                    shutil_rmtree = __import__("shutil").rmtree
                    shutil_rmtree(work.root, ignore_errors=True)

    def test_create_does_not_register_signal_handlers(self) -> None:
        with mock.patch(
            "tools.capture.workdir.find_tmpfs_base",
            return_value=Path(tempfile.gettempdir()),
        ):
            with mock.patch("tools.capture.workdir.atexit.register") as atexit_register:
                with mock.patch("signal.signal") as signal_signal:
                    work = CaptureWorkdir.create(
                        provider="claude",
                        allow_tmpfs_fallback=True,
                    )
                    try:
                        atexit_register.assert_called_once_with(work.remove_all)
                        signal_signal.assert_not_called()
                    finally:
                        if work.root.exists():
                            work.remove_all()


class FlowFlushTests(unittest.TestCase):
    def test_wait_for_flow_flush_skips_zero_grace(self) -> None:
        with mock.patch("tools.capture.runner.time.sleep") as sleep:
            _wait_for_flow_flush(0)
        sleep.assert_not_called()

    def test_wait_for_flow_flush_sleeps_when_positive(self) -> None:
        with mock.patch("tools.capture.runner.time.sleep") as sleep:
            _wait_for_flow_flush(0.25)
        sleep.assert_called_once_with(0.25)


class SignalCleanupTests(unittest.TestCase):
    def test_capture_signal_cleanup_stops_mitm_before_remove_all(self) -> None:
        work = mock.Mock()
        calls: list[str] = []

        def track_stop(_pid: int) -> None:
            calls.append("stop_mitm")

        def track_remove() -> None:
            calls.append("remove_all")

        work.remove_all.side_effect = track_remove
        with mock.patch("tools.capture.runner._stop_mitm", side_effect=track_stop):
            _capture_signal_cleanup(work, 4242)
        self.assertEqual(calls, ["stop_mitm", "remove_all"])
        work.remove_all.assert_called_once_with()

    def test_install_handler_stops_mitm_before_remove_all(self) -> None:
        work = mock.Mock()
        calls: list[str] = []

        def track_stop(_pid: int) -> None:
            calls.append("stop_mitm")

        def track_remove() -> None:
            calls.append("remove_all")

        work.remove_all.side_effect = track_remove
        previous = {signal.SIGINT: signal.SIG_DFL}
        with mock.patch("tools.capture.runner._stop_mitm", side_effect=track_stop):
            with mock.patch("signal.signal", return_value=signal.SIG_DFL):
                with mock.patch("os.kill") as kill:
                    _install_capture_signal_handlers(work, 4242)
                    handler = signal.signal.call_args[0][1]
                    handler(signal.SIGINT, None)
        self.assertEqual(calls, ["stop_mitm", "remove_all"])
        kill.assert_called_once_with(os.getpid(), signal.SIGINT)

    def test_restore_capture_signal_handlers(self) -> None:
        sentinel = object()
        with mock.patch("signal.signal") as signal_signal:
            _restore_capture_signal_handlers({signal.SIGTERM: sentinel})
        signal_signal.assert_called_once_with(signal.SIGTERM, sentinel)


class RunnerDryRunTests(unittest.TestCase):
    def test_dry_run_all_providers_and_modes(self) -> None:
        with mock.patch(
            "tools.capture.runner.stage_credentials",
            side_effect=AssertionError("stage_credentials must not run in dry-run"),
        ):
            with mock.patch(
                "tools.capture.runner.pick_free_port",
                side_effect=AssertionError("pick_free_port must not run in dry-run"),
            ):
                for provider in ("claude", "grok", "codex"):
                    for mode in ("general", "refresh"):
                        with self.subTest(provider=provider, mode=mode):
                            result = run_capture(
                                provider=provider,
                                mode=mode,
                                dry_run=True,
                            )
                            self.assertTrue(result["dry_run"])
                            plan = result["plan"]
                            self.assertEqual(plan["provider"], provider)
                            self.assertEqual(plan["port"], 0)
                            self.assertNotIn("workdir", result)

    def test_dry_run_uses_placeholder_paths(self) -> None:
        staged = plan_credentials(provider="grok", mode="general")
        self.assertEqual(str(staged.clean_home), "<clean-home>")
        self.assertTrue(str(staged.copied_paths[0]).startswith("<"))

    def test_dry_run_leaves_no_workdirs(self) -> None:
        before = set(Path(tempfile.gettempdir()).glob("omni-*-capture.*"))
        run_capture(provider="claude", mode="general", dry_run=True)
        after = set(Path(tempfile.gettempdir()).glob("omni-*-capture.*"))
        self.assertEqual(before, after)


class RunnerLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fake_home = Path(self.temp.name) / "source-home"
        self.fake_home.mkdir()
        claude_dir = self.fake_home / ".claude"
        claude_dir.mkdir()
        (claude_dir / ".credentials.json").write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "x", "expiresAt": 9}}),
            encoding="utf-8",
        )

    def _mock_workdir(self, *, keep_flow: bool = False) -> CaptureWorkdir:
        root = Path(self.temp.name) / "work"
        root.mkdir()
        clean_home = root / "clean-home"
        clean_home.mkdir()
        clean_codex = root / "clean-codex-home"
        clean_codex.mkdir()
        return CaptureWorkdir(
            base=root.parent,
            root=root,
            flow_path=root / "claude-capture.flow",
            extract_path=root / "claude-capture-extract.md",
            clean_home=clean_home,
            clean_codex_home=clean_codex,
            keep_flow=keep_flow,
        )

    def _run_successful_capture(self, work: CaptureWorkdir) -> dict[str, object]:
        completed = subprocess.CompletedProcess(args=["claude"], returncode=0)

        def fake_run(*_args, **_kwargs):
            return completed

        with mock.patch("tools.capture.runner._pgrep_available", return_value=True):
            with mock.patch("tools.capture.runner.require_mitmproxy_flow_reader"):
                with mock.patch("tools.capture.runner._wait_for_port"):
                    with mock.patch("tools.capture.runner._stop_mitm"):
                        with mock.patch("tools.capture.runner.subprocess.Popen") as popen:
                            popen.return_value.pid = 1234
                            with mock.patch(
                                "tools.capture.runner.subprocess.run",
                                side_effect=fake_run,
                            ):
                                with mock.patch(
                                    "tools.capture.runner.hosts_in_flow_file",
                                    return_value={"api.anthropic.com"},
                                ):
                                    with mock.patch(
                                        "tools.capture.runner.extract_flow_markdown",
                                    ) as extract:

                                        def _write_extract(*_args, **kwargs):
                                            out = kwargs.get("out")
                                            if out is not None:
                                                out.write("# extract\n")
                                            return 0

                                        extract.side_effect = _write_extract
                                        with mock.patch(
                                            "tools.capture.runner.stage_credentials",
                                            return_value=stage_credentials(
                                                provider="claude",
                                                clean_home=work.clean_home,
                                                clean_codex_home=None,
                                                mode="general",
                                                source_home=self.fake_home,
                                            ),
                                        ):
                                            return run_capture(
                                                provider="claude",
                                                mode="general",
                                                live_flag=True,
                                                workdir=work,
                                            )

    def test_missing_flow_reader_fails_before_provider_commands(self) -> None:
        work = self._mock_workdir()

        with mock.patch("tools.capture.runner._pgrep_available", return_value=True):
            with mock.patch(
                "tools.capture.runner.require_mitmproxy_flow_reader",
                side_effect=RuntimeError("FlowReader unavailable"),
            ):
                with mock.patch("tools.capture.runner.subprocess.Popen") as popen:
                    with mock.patch(
                        "tools.capture.runner.subprocess.run",
                        side_effect=AssertionError("provider commands must not run"),
                    ):
                        with mock.patch(
                            "tools.capture.runner.stage_credentials",
                            return_value=stage_credentials(
                                provider="claude",
                                clean_home=work.clean_home,
                                clean_codex_home=None,
                                mode="general",
                                source_home=self.fake_home,
                            ),
                        ):
                            with self.assertRaises(CaptureError) as ctx:
                                run_capture(
                                    provider="claude",
                                    mode="general",
                                    live_flag=True,
                                    workdir=work,
                                )
        popen.assert_not_called()
        self.assertIn("FlowReader unavailable", str(ctx.exception))

    def test_missing_mitm_ca_fails_before_provider_commands(self) -> None:
        work = self._mock_workdir()
        grok_dir = self.fake_home / ".grok"
        grok_dir.mkdir(exist_ok=True)
        (grok_dir / "auth.json").write_text(
            '{"https://auth.x.ai::x":{"key":"jwt","auth_mode":"oidc","expires_at":"2999-01-01T00:00:00Z"}}',
            encoding="utf-8",
        )

        with mock.patch("tools.capture.runner._pgrep_available", return_value=True):
            with mock.patch("tools.capture.providers.mitm_ca_path", return_value=None):
                with mock.patch("tools.capture.runner.subprocess.Popen") as popen:
                    with mock.patch(
                        "tools.capture.runner.subprocess.run",
                        side_effect=AssertionError("provider commands must not run"),
                    ):
                        with mock.patch(
                            "tools.capture.runner.stage_credentials",
                            return_value=stage_credentials(
                                provider="grok",
                                clean_home=work.clean_home,
                                clean_codex_home=None,
                                mode="general",
                                source_home=self.fake_home,
                            ),
                        ):
                            with self.assertRaises(CaptureError) as ctx:
                                run_capture(
                                    provider="grok",
                                    mode="general",
                                    live_flag=True,
                                    workdir=work,
                                )
        popen.assert_not_called()
        self.assertIn("mitmproxy CA cert", str(ctx.exception))

    def test_provider_command_nonzero_is_fatal(self) -> None:
        work = self._mock_workdir()
        completed = subprocess.CompletedProcess(args=["claude"], returncode=1)

        def fake_run(*_args, **_kwargs):
            return completed

        with mock.patch("tools.capture.runner._pgrep_available", return_value=True):
            with mock.patch("tools.capture.runner.require_mitmproxy_flow_reader"):
                with mock.patch("tools.capture.runner._wait_for_port"):
                    with mock.patch("tools.capture.runner._stop_mitm"):
                        with mock.patch("tools.capture.runner.subprocess.Popen") as popen:
                            popen.return_value.pid = 1234
                            with mock.patch(
                                "tools.capture.runner.subprocess.run",
                                side_effect=fake_run,
                            ):
                                with mock.patch(
                                    "tools.capture.runner.stage_credentials",
                                    return_value=stage_credentials(
                                        provider="claude",
                                        clean_home=work.clean_home,
                                        clean_codex_home=None,
                                        mode="general",
                                        source_home=self.fake_home,
                                    ),
                                ):
                                    with self.assertRaises(CaptureError) as ctx:
                                        run_capture(
                                            provider="claude",
                                            mode="general",
                                            live_flag=True,
                                            workdir=work,
                                        )
        self.assertIn("exit 1", str(ctx.exception))
        self.assertFalse(work.root.exists())

    def test_success_removes_staged_credentials(self) -> None:
        work = self._mock_workdir()
        work.flow_path.write_bytes(b"flow")
        result = self._run_successful_capture(work)
        self.assertEqual(result["provider"], "claude")
        self.assertEqual(result["mode"], "general")
        self.assertEqual(result["extract_text"], "# extract\n")
        self.assertEqual(result["captured_hosts"], ["api.anthropic.com"])
        self.assertIn("expected_hosts", result)
        self.assertNotIn("workdir", result)
        self.assertNotIn("flow_path", result)
        self.assertNotIn("extract_path", result)
        self.assertFalse(work.root.exists())

    def test_success_keep_flow_includes_paths(self) -> None:
        work = self._mock_workdir(keep_flow=True)
        work.flow_path.write_bytes(b"flow")
        result = self._run_successful_capture(work)
        self.assertEqual(result["extract_text"], "# extract\n")
        self.assertEqual(result["workdir"], str(work.root))
        self.assertEqual(result["flow_path"], str(work.flow_path))
        self.assertEqual(result["extract_path"], str(work.extract_path))
        self.assertTrue(work.root.exists())


class RefreshHelperTests(unittest.TestCase):
    def test_force_codex_preserves_refresh_token_expiry(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            auth = {
                "tokens": {
                    "access_token": "t",
                    "expires_at": "2999-01-01T00:00:00Z",
                    "expires_in": 3600,
                    "refresh_token_expires_at": "2999-06-01T00:00:00Z",
                },
                "last_refresh": "2999-01-01T00:00:00Z",
            }
            handle.write(json.dumps(auth))
            path = Path(handle.name)
        try:
            force_codex_expiry_stale(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                data["tokens"]["refresh_token_expires_at"],
                "2999-06-01T00:00:00Z",
            )
            self.assertEqual(data["tokens"]["expires_at"], "2000-01-01T00:00:00.000000Z")
            self.assertEqual(data["tokens"]["expires_in"], 0)
            self.assertEqual(data["last_refresh"], "2000-01-01T00:00:00.000000Z")
        finally:
            path.unlink(missing_ok=True)

    def test_force_helpers_reject_malformed_json(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write('{"bad": true}')
            path = Path(handle.name)
        try:
            with self.assertRaises(CredentialError):
                force_claude_expiry_stale(path)
            with self.assertRaises(CredentialError):
                force_grok_expiry_stale(path)
            with self.assertRaises(CredentialError):
                force_codex_expiry_stale(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
