# Claude Rebaseline Procedure

Use this when a new Claude Code release appears or a current pin is rejected
upstream.

## Breaking change (issue #12)

Omni ships **one live pin** per provider. Multi-version CLI flags
(`--claude-version`, `--grok-version`, `--codex-version`,
`--match-system`, `--match-system-exact`) and env counterparts
(`OMNI_*_VERSION`, `OMNI_MATCH_SYSTEM*`) are removed. Rebaseline **overwrites**
the single pin; it does not append a profile ladder. For older wire fingerprints,
use an older Omni release from git history.

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
- Historical cch algorithm (not a live step):
  `docs/providers/claude/CCH_ALGORITHM.md`

Live extraction requires the `mitmproxy` Python package in the same interpreter
as `tools.capture` (for example `uv run --with mitmproxy python -m tools.capture`).

## Procedure

1. Detect drift:

   ```sh
   uv run --script tools/providers/claude/fingerprint/check_claude_code_drift.py
   ```

   Continue only if `status` is not `ok`, or if a provider rejection requires a
   fresh capture despite a matching version.

2. Capture live traffic on tmpfs (requires `OMNI_CAPTURE_LIVE=1` or `--live-capture`):

   ```sh
   uv run --with mitmproxy python -m tools.capture capture run \
     --provider claude --mode general --live-capture \
     --models claude-mythos-5-2 claude-mythos-5-1 fable \
       claude-fable-5-1 claude-fable-5 opus sonnet haiku
   ```

   The legacy wrapper remains for compatibility:

   ```sh
   tools/providers/claude/fingerprint/capture_baseline.sh \
     claude-mythos-5-2 claude-mythos-5-1 claude-fable-5-2 \
     claude-fable-5-1 claude-fable-5 claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5
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
   uv run --with mitmproxy python -m tools.capture capture run \
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
   - Obtain this pin's catalog from captured `POST /v1/messages` model ids
     (`python3 -m tools.capture catalog --provider claude --flow-file ...`).
     If that listing is empty, stop. Do not keep the previous pin's catalog.
   - Confirm all pinned catalog models are accepted.
   - Confirm the billing suffix and no-cch header (or captured cch shape if
     a newer CLI reintroduces `cch=`).
   - If a newer Claude Code reintroduces `cch=`, stop. The live cch rewrite
     no longer exists in the crates; restore it from git history as a starting
     point (removed in 243b7ce, so `243b7ce^` holds the last live version) and
     re-derive the current algorithm with the reverse-engineering playbook in
     `docs/providers/claude/CCH_ALGORITHM.md`.

4. Update code (overwrite the single active pin; do not append a profile ladder):

   - Overwrite the active `FingerprintProfile` in
     `crates/provider-claude/src/fingerprint.rs` with the new capture.
   - Overwrite the single model catalog in `crates/provider-claude/src/models.rs`
     from this pin's listing only.
   - Rename shared constants if the pin version changes (do not keep historical
     `*_VERSION` symbol names that disagree with the live pin).
   - Update active-pin goldens. History lives in git tags and older Omni
     releases, not in-tree multi-profile selection.

5. Vector regeneration is not a live step. Historical cch lives in
   `docs/providers/claude/CCH_ALGORITHM.md`. The drift checker verifies
   version + no-cch billing header + suffix only.

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

- Drift checker agrees with the pinned version and no-cch billing header.
- Captured fields are represented in source.
- Default workspace tests pass without credentials or network.

## Current Capture

The 2026-11-09 clean-HOME capture used Claude Code 2.1.321. All ten
requests (default, Mythos 5.2, Mythos 5.1, explicit Fable 5.2, `fable`,
Fable 5.1, Fable 5, Opus, Sonnet, and Haiku) returned HTTP 200, completed
with `message_stop` and `end_turn`, and replied `OK`.

The `fable` alias resolves to Fable 5.2. Omni's `mythos` alias now selects
Mythos 5.2. Both 5.2 models have captured defaults of 128000 output tokens
and medium effort, with the same beta list as Opus. Mythos 5.1 and explicit
Fable 5 omit `per-turn-control-2026-07-01`; Fable 5.1 retains it. The older
Fable/Mythos defaults remain 64000/high. Explicit Fable 5 remains an
unadvertised pass-through target with freshly confirmed overrides.

The SDK, runtime, Anthropic version, system identity, other model defaults,
and no-cch billing shape remain unchanged. Exact captured header bytes are
recorded in `tools/providers/claude/fingerprint/BASELINE_HEADERS.md` and the
fingerprint golden tests. Catalog and defaults are defined in
`crates/provider-claude/src/models.rs` and
`crates/provider-claude/src/fingerprint.rs`.

The rebuilt isolated Omni server completed all three inbound APIs, with
streaming and non-streaming requests, for every advertised Claude model
and explicit Fable 5 (48 calls). The `fable` and `mythos` aliases also
completed Chat Completions requests. Both 5.2 models returned positive
native token counts. These checks prove basic request/response acceptance;
tool, media, and cache behavior were not live-tested in this pass.
