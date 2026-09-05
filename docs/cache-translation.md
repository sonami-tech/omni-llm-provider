# Cache translation

Status: shipped. Official inbound cache fields become internal cache intent
(`CanonicalRequest.cache` plus block/tool marks). Outbound backends receive
that backend's official cache fields only. Canonical types stay internal.

Official provider docs remain the source of truth for field names and
supported values. This file records Omni gateway decisions only.

## Goal

Any client on any inbound interface, using that interface’s **official** cache
fields, must get a successful request and the best cache the chosen backend
can provide. Omni does not invent a client-facing cache dialect.

Compatibility wins over exact TTL/mode fidelity. Do not 400 a client because
its official cache fields cannot be copied exactly.

## Terminology

**Providers:** `claude`, `grok`, `codex`

**Inbound interfaces:**

- Chat Completions `POST /v1/chat/completions`
- Responses `POST /v1/responses`
- Anthropic Messages `POST /v1/messages` (dual-mode)

Canonical types are internal only.

## Internal shape

Inbound parsers store **intent**, not leftover JSON.

Request-level object (`CanonicalCacheIntent`):

- `routing_identity` — from Chat/Responses `prompt_cache_key`, or Chat header
  `x-grok-conv-id` when no body key is present
- `mode` — OpenAI `prompt_cache_options.mode` (`implicit` | `explicit`)
- `ttl` — requested lifetime when the client sent one
- `legacy_retention` — OpenAI `prompt_cache_retention` (`in_memory` | `24h`)
- `automatic` — Anthropic top-level `cache_control`

Block- and tool-level marks stay on content and `CanonicalTool`. Unmarked text
may stay `CanonicalContent::Text`. Marked blocks must not collapse to a string.

Do not map `user`, `safety_identifier`, or Anthropic `metadata.user_id` to
routing identity.

## Inbound (official fields only)

**Chat Completions**

- Body `prompt_cache_key`
- Header `x-grok-conv-id` (xAI Chat clients)
- `prompt_cache_options`, `prompt_cache_retention`
- `prompt_cache_breakpoint` on official Chat parts

**Responses**

- Body `prompt_cache_key`
- `prompt_cache_options`, `prompt_cache_retention`
- `prompt_cache_breakpoint` on `input_text` / `input_image` / `input_file`
- No cache header is required
- Breakpoints forbidden on top-level `instructions` (put marked instructions
  in a developer `input_text` part)

**Anthropic Messages**

- Official `cache_control` only (top-level automatic, tools, system, content)
- TTL omit / `5m` / `1h`
- Reject body `prompt_cache_key` (not an Anthropic field; do not add an Omni
  dialect on this interface)

### Routing identity merge

- Header only → `routing_identity`
- Body `prompt_cache_key` only → `routing_identity`
- Both present → use `prompt_cache_key`, ignore the header. Do not 400. Do
  not forward both.

Native xAI Chat Completions accepts `prompt_cache_key` alone. Sending the
header and the body key together broke cache on a live isolated probe
(`/tmp/rc-cache-spike/SUMMARY.md`).

## Outbound

Always emit the backend’s official fields. Never emit `x-grok-conv-id`.

| Backend | Routing identity | Breakpoints / automatic | TTL |
|---|---|---|---|
| Claude | Consume; do not emit; do not 400 | `cache_control` (client marks beat gateway auto-cache) | `5m` (default) or `1h` |
| Grok Chat | Body `prompt_cache_key` | Native prefix cache; do not invent Grok breakpoint fields | None (drop) |
| Grok Responses (CLI) | Body `prompt_cache_key` | Same prefix cache | None (drop) |
| Codex REST and ChatGPT WS | Body `prompt_cache_key` (same payload) | `prompt_cache_breakpoint` on content; tool marks are prefix hints only | `30m` on GPT-5.6+; `prompt_cache_retention` on earlier models |

Grok `explicit` mode has no equivalent: keep the request, use Grok prefix
cache, drop the mode.

Same-provider passthrough: if the inbound value is already legal on the
chosen backend, send it unchanged. Clamp only when translating across
mismatched enums.

## Breakpoints and mode (locked)

**Decision:** Claude slot cap. Anthropic allows at most 4 cache slots,
including automatic. If translation would exceed 4, keep the last 4 marks
and drop the rest. Do not 400.

**Decision:** OpenAI `prompt_cache_options.mode` `explicit` with no
breakpoints, routed to Claude: do not inject gateway auto-cache. The client
asked not to cache.

**Decision:** Invalid values on the inbound dialect itself are 400.
Example: Anthropic `ttl: "30m"` is not a legal Anthropic field. Clamp-down
applies only when the inbound value is legal for that interface and the
chosen backend cannot copy it.

Claude fingerprint and billing header injection must stay in a stable prefix.
If Omni rewrites `system` or `tools` after the client’s cache marks, cache will
miss even when field translation is correct.

## Compatibility vs fail-loud

Gateway philosophy (`docs/README.md`, `docs/decisions.md`) is fail-loud when
a backend cannot honor intent that changes the end result.

**Exception for cache TTL and cache mode:** the client is not under our
control. 400 on unmappable TTL/mode makes Omni unusable. For those fields,
degrade to the backend’s native cache and complete the request.

Still 400 for:

- Invalid types (non-string key, bad enum literals on the *inbound* dialect)
- Breakpoint on an inbound location that dialect forbids (Responses
  `instructions`)
- Anthropic body `prompt_cache_key`

## TTL (locked)

**Decision:** clamp-down. Pick the longest official backend value that is
**less than or equal to** the requested duration. Do not 400. Do not round
up. Do not reset to the backend default when a longer-but-still-≤ value
exists.

These are discrete enums, not a continuous range. Official values (provider
docs, not Omni):

- Anthropic: default/`5m`, optional `1h`
- OpenAI GPT-5.6+: `prompt_cache_options.ttl` is only `30m` (also the default)
- OpenAI earlier models: `prompt_cache_retention` `in_memory` (~5–10 minutes,
  max 1 hour) or `24h`
- xAI: no TTL API

Examples:

- Client asks `1h`, backend offers `5m` and `30m` → send `30m`, not `5m`.
- Client asks `30m`, Claude offers `5m` and `1h` → send `5m`, not `1h`
  (`1h` would retain longer than asked; Claude has no `30m`).

If the backend has no value ≤ the request (client asks `5m`, Codex GPT-5.6+
only has `30m`), omit TTL and use that backend’s native/default cache. The
request still succeeds.

Grok has no TTL field. Drop it. Prefix cache duration is Grok’s.

## Current tree vs this spec

Shipped. The interim `CanonicalRequest.prompt_cache_key: Option<String>` lift
is gone. Inbound parsers store `CanonicalCacheIntent` plus block/tool marks.
Grok Chat outbound sends body `prompt_cache_key` and never `x-grok-conv-id`.
Anthropic inbound `prompt_cache_key` is 400; `cache_control` is translated.

## Tests and live policy

Hermetic tests cover parse, merge, translate, and “key is not an extra.”

Live cache tests are opt-in (`OMNI_LIVE_TESTS=1`) only. Do not hardcode dated
model ids: catalog first id, `current_model()`, or documented aliases
(`haiku`), never a pin.

Do not run live tests unless the operator approves quota. Do not commit unless
asked.

## Sources

- OpenAI prompt caching: `prompt_cache_key`, `prompt_cache_options`,
  `prompt_cache_breakpoint`, `prompt_cache_retention`
- xAI prompt caching: automatic prefix cache; Chat `x-grok-conv-id`;
  Responses `prompt_cache_key`
- Anthropic prompt caching: `cache_control`, automatic + explicit, 4 slots,
  TTL `5m`/`1h`
- Live affinity probe: `/tmp/rc-cache-spike/SUMMARY.md`
