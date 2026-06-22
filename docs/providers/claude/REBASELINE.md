# Claude Rebaseline Procedure

Use this when a new Claude Code release appears or a current profile is rejected
upstream.

## Safety

- Raw mitmproxy `.flow` files contain live OAuth bearer tokens.
- Keep raw flows on RAM-backed tmpfs only.
- Never commit `.flow` files, credentials, extracted bearer tokens, or local
  reports containing unredacted auth.
- Use clean HOME/CWD captures so project or user instruction files are not
  copied into request bodies or reports.
- Do not run live capture without explicit operator approval.
- Refresh capture additionally mutates only the staged credential copy to force
  expiry. Never edit the real credential file for capture.

## Tools

- Shared capture CLI: `python3 -m tools.capture` (source of truth for live
  capture, refresh capture, staging, MITM, extraction, and cleanup)
- `tools/providers/claude/fingerprint/check_claude_code_drift.py`
- `tools/providers/claude/fingerprint/capture_baseline.sh` (thin wrapper; prefer the shared CLI for new work)
- `tools/providers/claude/fingerprint/extract_flow.py` (compatibility wrapper around `tools.capture extract flow`)
- `tools/providers/claude/fingerprint/BASELINE_HEADERS.md`
- `tools/providers/claude/fingerprint/CCH_ALGORITHM.md`
- `tools/providers/claude/fingerprint/vectors/`

## Procedure

1. Detect drift:

   ```sh
   uv run --script tools/providers/claude/fingerprint/check_claude_code_drift.py
   ```

   Continue only if `status` is not `ok`, or if a provider rejection requires a
   fresh capture despite a matching version.

2. Capture live traffic on tmpfs (requires `OMNI_CAPTURE_LIVE=1` or `--live-capture`):

   ```sh
   python3 -m tools.capture capture run \
     --provider claude --mode general --live-capture \
     --models claude-fable-5 claude-haiku-4-5 claude-sonnet-4-6 claude-opus-4-8
   ```

   The legacy wrapper remains for compatibility:

   ```sh
   tools/providers/claude/fingerprint/capture_baseline.sh \
     claude-fable-5 claude-haiku-4-5 claude-sonnet-4-6 claude-opus-4-8
   ```

   Both helpers start mitmdump as a reverse proxy to `https://api.anthropic.com`,
   copy only Claude credentials into a clean tmpfs HOME, drive the installed
   `claude` CLI from that clean HOME/CWD, extracts a redacted structural
   Markdown report, and removes the tmpfs workdir (including staged credential
   copies) unless `KEEP_FLOW=1`. `KEEP_FLOW=1` retains the workdir and raw flow
   on tmpfs and prints warnings.

   Refresh validation proves Anthropic API-host traffic through the reverse
   proxy. Separate auth-host proof awaits a stable observed auth endpoint.

   Refresh capture command:

   ```sh
   python3 -m tools.capture capture run \
     --provider claude --mode refresh --live-capture --refresh-capture
   ```

3. Analyze the extract:

   - Confirm `POST /v1/messages?beta=true`.
   - Record send-order headers.
   - Compare `anthropic-beta`, stainless package/runtime versions, and
     `anthropic-version`.
   - Confirm `model`, `max_tokens`, `temperature`, `thinking` /
     `output_config`, `metadata`, `context_management`, `stream`, and system
     block structure.
   - Confirm default model from the no-`--model` capture.
   - Confirm all pinned catalog models are accepted.
   - Confirm the billing suffix and cch behavior.
   - If any checksum or body mutation cannot be reproduced exactly, do not
     promote the profile to `latest`.

4. Update code:

   - Add or update the `FingerprintProfile` in
     `crates/provider-claude/src/fingerprint.rs`.
   - Add or update model catalog entries in `crates/provider-claude/src/models.rs`.
   - Update the default/latest profile only after the new profile is proven.
   - Update tests and local vectors.

5. Regenerate clean-room cch vectors:

   ```sh
   uv run --script tools/providers/claude/fingerprint/check_claude_code_drift.py \
     --emit-vectors tools/providers/claude/fingerprint/vectors
   ```

6. Update docs:

   - `tools/providers/claude/fingerprint/BASELINE_HEADERS.md`
   - `docs/providers/claude/README.md` only if structure or invariant changed.
   - `docs/providers/README.md` only if shared capture policy changed.

7. Verify:

   ```sh
   cargo test -p provider-claude
   cargo test --workspace
   cargo clippy --workspace --all-targets -- -D warnings
   ```

8. Optional live smoke, only with approval:

   ```sh
   OMNI_LIVE_TESTS=1 cargo test -p provider-claude claude_send_exercises_full_fingerprint_path
   ```

## Done Criteria

- Drift checker agrees with the pinned version and cch.
- Captured fields are represented in source.
- Recovered vectors are local to this repo and covered by Rust tests.
- Default workspace tests pass without credentials or network.

## Current 2.1.175 Status

On 2026-06-12, Claude Code 2.1.175 was captured and model behavior was verified
for default, `fable`, `opus`, `sonnet`, and `haiku` flows. Headers still use SDK
package `0.94.0`, runtime `v24.3.0`, Anthropic version `2023-06-01`, and
`claude-cli/2.1.175 (external, sdk-cli)`.

2.1.175 is the current `latest` profile. Its cch path is still xxHash64 seeded
with `0x4d659218e32a3268`, but the hashed input removes model string values and
the numeric `max_tokens` field before hashing. The transform is covered by
clean-room vectors for Fable, Opus, Sonnet, and Haiku.
