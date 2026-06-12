# Claude Code cch fingerprint

Canonical target: Claude Code `2.1.165`, profile `cc-2.1.165-sdk-cli`.
The same checksum algorithm remains verified for `cc-2.1.142-sdk-cli`,
`cc-2.1.150-sdk-cli`, `cc-2.1.154-sdk-cli`, `cc-2.1.158-sdk-cli`,
`cc-2.1.161-sdk-cli`, and `cc-2.1.162-sdk-cli`.

This document is the repo-canonical record for the `cch` field in Claude
Code's billing marker. If this conflicts with an agent memory note, update the
memory to point here.

## Wire behavior

Claude Code first builds this system text block:

```text
x-anthropic-billing-header: cc_version=2.1.158.175; cc_entrypoint=sdk-cli; cch=00000;
```

Before the final HTTP request leaves the process, Claude Code rewrites the five
zeroes. The rewrite is deterministic for the exact request-body bytes.

Algorithm recovered for `2.1.142` and re-verified for `2.1.150`, `2.1.154`, and
`2.1.158` through `2.1.165`:

1. Serialize the final `/v1/messages` JSON request body with `cch=00000;`
   still present.
2. Compute standard `xxHash64(body_bytes, seed = 0x4d659218e32a3268)`.
3. Take `hash & 0xfffff`.
4. Format as five lowercase hex digits with zero padding.
5. Overwrite the five `00000` bytes in the billing header.

The body length does not change. A non-sentinel value such as `cch=abcde;` is
left untouched.

Omni computes the checksum over the exact JSON bytes Omni sends. Those bytes do
not have to match Claude Code's field order for the checksum to be internally
consistent, but any future attempt at byte-for-byte Claude Code body matching
must re-check serializer order.

## Omni implementation notes

- The active profile `cc-2.1.158-sdk-cli` owns the checksum behavior.
- The visible billing header still starts with `cch=00000;`; the final-body
  hook rewrites it immediately before logging/sending the upstream body.
- The rewrite targets the first matching billing header under the serialized
  `system` field, not a bare `cch=00000` substring. This avoids corrupting user
  message text that happens to contain the sentinel.
- `--no-preamble` or any body without the billing sentinel is left unchanged.
- Retries recompute from the original `serde_json::Value`; because this
  algorithm only depends on body bytes, logging and retry sends remain
  deterministic.

## Verified fixtures

The following captured Claude Code final bodies validate the seed and checksum.
To verify a row, replace the listed final `cch` value with `00000`, compute the
algorithm above, and compare.

| Scenario | Expected cch |
|---|---:|
| minimal `Say OK` body | `3bc55` |
| factor body with Claude preamble | `9bce0` |
| system field serialized before model/messages | `4dc19` |
| two billing markers; first sentinel rewritten | `7afbb` |
| watchpoint marker body | `c159b` |

Rust unit tests in `crates/provider-claude/src/fingerprint.rs` lock these fixtures in.

## Reverse-engineering playbook

Use this if a future Claude Code release changes the behavior:

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
server, reports the installed version, and verifies whether the observed final
body still matches the pinned checksum algorithm.

## 2.1.175 Status

Claude Code 2.1.175 still exposes a `cch=00000` marker before transport, but
the final request body observed on 2026-06-12 does not match this xxHash64 body
algorithm. Do not reuse this algorithm for 2.1.175 or promote that release until
the new checksum path is recovered and pinned with vectors.
