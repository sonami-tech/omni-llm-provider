# Grok Provider

Grok-specific behavior lives in `crates/provider-grok`.

## Source Of Truth

- xAI mapping and streaming parser: `crates/provider-grok/src/lib.rs`
- Credential loader: `crates/provider-grok/src/credentials.rs`
- Gate notes: `docs/grok-gate.md`
- Capture procedure: `docs/providers/grok/CAPTURE.md`
- Shared capture framework: `tools/capture/`
- Compatibility capture wrappers: `tools/providers/grok/capture/`

## Invariant

Grok does not currently require Claude-style billing headers or injected
identity preambles. The maintained contract is:

- CLI-parity wire to `cli-chat-proxy.grok.com` (`POST /v1/responses`),
- fresh credential resolution per request (prefers `~/.grok/auth.json` OIDC),
- grok-shell fingerprint headers for the pinned CLI version,
- correct non-stream and stream decoding via the shared Responses parser,
- model catalog kept current; the active version and catalog are defined in
  `crates/provider-grok/src/lib.rs`. Catalog source is clean-HOME
  `GET /v1/models` on `cli-chat-proxy.grok.com`. See
  [Catalog exception](#catalog-exception).

`--grok-version` / `OMNI_GROK_VERSION` and match-system flags are removed
(issue #12). Rebaseline overwrites this pin; older wire needs an older Omni
release.

Default path credentials: `$XAI_CREDENTIALS_PATH`, then `~/.grok/auth.json`, then
`~/.xai/.credentials.json`. Custom endpoint mode is different by design.
`OMNI_GROK_BASE_URL` is Omni's forced override and uses only
`OMNI_GROK_AUTH_TOKEN`, `OMNI_GROK_API_KEY`, and `OMNI_GROK_CUSTOM_HEADERS`.
Legacy `GROK_MODELS_BASE_URL` remains supported and uses `XAI_API_KEY` per
request only, or no Authorization header if it is unset. Default CLI credentials
must not be sent to a custom endpoint.

Default tests use wiremock. Live calls require `OMNI_LIVE_TESTS=1`.

Capture and refresh-capture work uses `python3 -m tools.capture`; see
`docs/providers/grok/CAPTURE.md`.

## Catalog exception

The 2026-10-15 grok-shell 1.0.67 clean-HOME capture listed only `grok-4.7`.
The capture account is on the Free tier with a temporary Grok 4.7 promo.
The default inference request completed with HTTP 200 after one CLI-managed
429 retry. Background session-title requests to `grok-4.6` returned 429;
this run does not prove that model's current acceptance. Request header
names and body shape match the previous pin apart from the CLI version.

The operator renewed the issue #56 exception for this pin: keep the four
models from the 1.0.41 listing. Source of truth is `GROK_CATALOG` in
`crates/provider-grok/src/lib.rs`. At the next rebaseline, take the listing
from a paid-tier account, or obtain a new operator decision to keep the
exception.

## Reasoning effort

Maps explicit client effort to xAI `low|medium|high`, plus `xhigh` on
`grok-4.7`, `grok-4.7-build-fast`, and `grok-4.6` (aliases `minimal`→`low`, `max`→`high`). Explicit `"none"` omits the
field. Unmappable values, including `xhigh` on `grok-4.5`, fail loud (issue #20).

**Omit when client omits (issue #18):** if the client does not set effort, Omni
does not send one. The provider/model default applies (often `high` on
grok-4.5). No force-floor and no invented disable. Capture notes:
`docs/providers/grok/CAPTURE.md`. Operator summary: `docs/README.md`.

## Parallel tool control

`parallel_tool_calls` is a typed canonical boolean, not a provider extra.
Chat and Responses treat missing/null as absent and reject other types.
Translated Messages inverts both explicit disabling booleans. Explicit values
are emitted unchanged and absent controls stay absent, including named
single-call requests and no-tools requests. Both custom-auth Chat and default
CLI Responses builders carry the control.
Streaming and non-streaming paths share these mappings. Offline tests prove
serialization only; actual CLI acceptance of explicit true/false is unverified.

## Provider Extras

Grok accepts these provider extras on OpenAI-compatible inbound surfaces:

- `service_tier`
- `search_parameters`
- `response_format`
- `seed`
- `stop`
- `n`
- `tools`

Unsupported extras fail loudly. `previous_response_id` is not forwarded on the
custom chat-completions override path.
