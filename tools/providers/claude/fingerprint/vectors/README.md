# Real-traffic cch capture vectors

Each `vector-<version>-<model>.json` in this directory is a **real Claude Code
request body** captured from the installed CLI, used as a recovered-capture
fixture for Omni's `x-anthropic-billing-header` cch checksum.

Rust tests in `crates/provider-claude/src/fingerprint.rs` load these via
`include_str!`, replace the embedded `cch=<value>` with the `cch=00000`
sentinel, recompute the profile's checksum, and assert it reproduces the
captured value. This proves Omni's body serialization + cch algorithm match real
Claude Code traffic **over a full-shape body** (`metadata`,
`context_management`, `thinking`, `tools`, `cache_control`, multiple `system`
blocks) - not just hand-minimized bodies and not Omni's own re-serialized output
(which would be circular).

## Why these are safe to commit

They are produced in a **clean room** (a credential-only temp `HOME` on tmpfs +
a clean CWD), so no project/global `CLAUDE.md` or skills are injected. The
generator then **normalizes** every volatile or host/account-identifying field
to a fixed placeholder and recomputes the cch over the normalized bytes:

| Field | Placeholder | Why |
|---|---|---|
| `metadata.user_id.session_id` | `00000000-0000-4000-8000-000000000000` | fresh per request |
| `metadata.user_id.device_id` | 64 x `0` | per-machine id |
| clean `HOME` path (3 encodings) | `/home/omni-vector` | random + uid-bearing |
| account email | `vector@example.com` | from the OAuth identity |

The generator fails closed if a bearer token, an email address, or the
structural signal of a loaded instruction file (a `Contents of .../CLAUDE.md`
header) survives into a vector - so a future Claude Code version that re-injects
local instructions via a new path aborts the emit instead of leaking them.

## Regenerating (do this on every rebaseline)

```sh
uv run --script tools/providers/claude/fingerprint/check_claude_code_drift.py --emit-vectors tools/providers/claude/fingerprint/vectors
```

Output is **deterministic**: a fresh run produces byte-identical files (the
fixed clean-`HOME` path + field normalization remove all run-to-run variance).
After regenerating for a new version, update the `include_str!` paths and the
`<version>` references in the Rust test, then `cargo test`. Do **not** hand-edit
these files - the embedded cch must stay consistent with the body bytes, which
only the generator guarantees.
