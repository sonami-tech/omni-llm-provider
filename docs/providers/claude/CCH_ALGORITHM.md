# Claude Code cch fingerprint (historical)

**Historical.** Live Omni no longer computes or rewrites `cch=`. The active pin
(Claude Code `2.1.259`, captured 2026-09-03) emits a billing header that ends at
`cc_entrypoint=sdk-cli;` with no checksum field. The `cc_version` suffix
(`Sha256Utf16SampleV1`) is still live.

This document records the `cch` rewrite algorithm used by earlier Claude Code
releases (`2.1.142` through `2.1.175`) and by retired clean-room vectors. It is
not a live rebaseline step. Rebaseline overwrites the single active pin; it does
not reintroduce a multi-profile ladder.

This document is the repo-canonical record for the historical `cch` field. If
this conflicts with an agent memory note, update the memory to point here.

## Wire behavior (historical)

Claude Code first built this system text block:

```text
x-anthropic-billing-header: cc_version=2.1.158.175; cc_entrypoint=sdk-cli; cch=00000;
```

Before the final HTTP request left the process, Claude Code rewrote the five
zeroes. The rewrite was deterministic for the exact request-body bytes.

Algorithm recovered for `2.1.142` and re-verified for `2.1.150`, `2.1.154`, and
`2.1.158` through `2.1.165`:

1. Serialize the final `/v1/messages` JSON request body with `cch=00000;`
   still present.
2. Compute standard `xxHash64(body_bytes, seed = 0x4d659218e32a3268)`.
3. Take `hash & 0xfffff`.
4. Format as five lowercase hex digits with zero padding.
5. Overwrite the five `00000` bytes in the billing header.

Algorithm recovered for `2.1.175`:

1. Serialize the final `/v1/messages` JSON request body with `cch=00000;`
   still present.
2. Remove every JSON `"model":"<value>"` string value while preserving
   `"model":""`.
3. Remove each numeric `"max_tokens":<n>` field plus one adjacent comma.
4. Compute standard `xxHash64(transformed_bytes, seed = 0x4d659218e32a3268)`.
5. Take `hash & 0xfffff`, format as five lowercase hex digits, and overwrite
   the five `00000` bytes.

The body length did not change. A non-sentinel value such as `cch=abcde;` was
left untouched.

## Historical Omni notes

These notes describe the retired compiled rewrite path. They are not live:

- The visible billing header started with `cch=00000;`; a final-body hook
  rewrote it immediately before logging/sending the upstream body.
- The rewrite targeted the first matching billing header under the serialized
  `system` field, not a bare `cch=00000` substring.
- `--no-preamble` or any body without the billing sentinel was left unchanged.
- Retries recomputed from the original `serde_json::Value`.

## Verified fixtures (historical)

The following captured Claude Code final bodies validated the seed and checksum.
To verify a row, replace the listed final `cch` value with `00000`, compute the
algorithm above, and compare.

| Scenario | Expected cch |
|---|---:|
| minimal `Say OK` body | `3bc55` |
| factor body with Claude preamble | `9bce0` |
| system field serialized before model/messages | `4dc19` |
| two billing markers; first sentinel rewritten | `7afbb` |
| watchpoint marker body | `c159b` |

Clean-room vectors for `2.1.162`, `2.1.165`, and `2.1.175` lived under
`tools/providers/claude/fingerprint/vectors/` and were deleted when live CCH
left the compiled crates. History is in git.

## Reverse-engineering playbook

Use this if a future Claude Code release reintroduces `cch=`:

1. Run a fake local Anthropic server and point Claude Code at it with
   `ANTHROPIC_BASE_URL`. Capture the final HTTP request bodies.
2. Add a `BUN_OPTIONS=--preload ...` hook that logs `fetch`, `JSON.stringify`,
   `crypto`, and `TextEncoder` inputs. For `2.1.142`, this proved the JS-visible
   body still had `cch=00000` while the final transport body had a nonzero cch.
3. If source inspection is insufficient, preload a unique marker into the body,
   write a ready file with PID/body metadata, and pause before fetch until a
   go-file exists.
4. If `gdb`/`lldb` is unavailable, use a parent `ptrace` helper that forks
   Claude Code, attaches all threads, scans writable mappings for the compact
   JSON body containing the marker and `cch=00000`, and sets a hardware
   watchpoint on the first checksum byte.
5. Enable clone/fork/vfork tracing and reapply watchpoints to new worker
   threads. In `2.1.142`, the mutation happened in a new worker thread.
6. On the watchpoint trap, record RIP, watched bytes, and disassemble nearby
   code with `objdump`.

For `2.1.142`, the trap was in the installed binary at `0x2e068ae`; the
preceding DWORD write at `0x2e068a8` wrote the first four hex digits, and the
next instruction wrote the fifth. Nearby code searched `/v1/messages`, searched
the `cch=00000` sentinel, initialized xxHash64 with seed
`0x4d659218e32a3268`, finalized, masked to 20 bits, and wrote five lowercase
hex chars.

## Drift checking

Normal unit and integration tests should not invoke the local Claude Code
installation because they would depend on live credentials, network access, and
whatever version happens to be installed.

Use the opt-in script instead:

```sh
tools/providers/claude/fingerprint/check_claude_code_drift.py
```

The script captures a live local Claude Code request against a fake Anthropic
server, reports the installed version, and verifies the no-cch billing header
plus `cc_version` suffix against the live pin. Vector regeneration is not a
live step.
