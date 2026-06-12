# Key Decisions and Findings

## Architecture

- Decision: one server binary (`omni`) with separate provider crates.
- Rationale:
  - Claude fingerprint logic is isolated in `provider-claude`.
  - Grok wire logic is isolated in `provider-grok`.
  - Shared HTTP conversion, Responses conversion, auth, stats, replacements,
    session derivation, conversation logging, and error envelopes live in
    `omni-common`.
  - `omni` only routes, frames responses, exposes catalogs, records stats, and
    wires optional conversation logging.

## Routing

- Prefix routing selects the backend: `claude:<model>` or `grok:<model>`.
- With exactly one provider enabled, bare model names are accepted.
- With multiple providers enabled, bare model names are rejected.

## Provider Boundaries

- Claude: cch, betas, preamble, profiles, model aliases, credentials, and
  Anthropic wire defaults stay in `provider-claude`.
- Grok: xAI request/response mapping, streaming parsing, credential resolution,
  and model catalog stay in `provider-grok`.
- Server concerns: auth, stats, bind/public flags, route registration, and model
  prefixing stay in `omni`.

## Current Surfaces

- OpenAI Chat Completions: `/v1/chat/completions`
- OpenAI Responses subset: `/v1/responses`
- Models: `/v1/models`, `/models`
- Stats: `/stats`
- Health/root: `/health`, `/`

## Credentials

Credentials are read fresh per request.

- Claude: `$CLAUDE_CREDENTIALS_PATH` or `~/.claude/.credentials.json`
- Grok: `$XAI_CREDENTIALS_PATH`, `~/.xai/.credentials.json`, or
  `~/.grok/auth.json`

## Tests

- Default tests are hermetic and must not call live providers.
- Live provider tests require `OMNI_LIVE_TESTS=1` plus usable credentials.
- Subprocess HTTP tests use shared Rust helpers in `omni-common::test_support`
  instead of shelling out to `curl`.
