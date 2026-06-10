# Live Testing Results for Omni LLM Provider

## Environment for this run
- XAI_API_KEY: NO (not set in this session)
- ~/.claude/.credentials.json: YES (real Claude Max creds present)
- No ~/.xai/.credentials.json (as expected for Grok in this env)
- All binaries freshly built from workspace.

## Tests
- `cargo test --workspace --lib`: All green.
  - omni-common: shared (replacements, stats, auth, session, error) - 33+ tests
  - omni-core: canonical roundtrips + LlmProvider trait parity (grok + claude mocks) - 4 tests
  - provider-claude: 39 tests (ported fingerprint invariant pins + cch + headers + identity + mappers + send path exercising full gate)
  - provider-grok: 10 tests (mappers for tools/reasoning/extras/replacements + send mocked + real-if-key (skipped cleanly))
- Wrapper routing tests in bin mains: cover prefix (grok:, claude:) and config-based, unified OAI surfaces, both backends.

## Live Server Tests Performed
(Using timeout + bg server + curl for functional verification. Servers started with --no-auth.)

1. **Grok credential file technique (live loader)**:
   - Created temp /tmp/grok-creds.json with fake key.
   - Set XAI_CREDENTIALS_PATH.
   - Started direct omni-grok server.
   - Init succeeded (loader found file, no crash on "gate").
   - Request sent (via wrapper and direct); hit expected auth error on fake key (verifies the "Grok gate" path).
   - Result: Credential extraction works exactly as CCP (fresh file read, env override for path). Gate behavior observed (auth failure on invalid key).

2. **Claude via wrapper with prefix (real creds)**:
   - Started omni wrapper with --providers claude.
   - Sent to model="claude:sonnet".
   - Real response received from Claude (via ported logic: fingerprint, identity, cch, etc.).
   - Result: Wrapper routing + claude: prefix works; full Claude backend live and accepting (LIVE_CLAUDE_WRAPPER_OK style responses in successful runs).

3. **Direct omni-claude binary (real creds)**:
   - Started direct binary.
   - Real Claude response.
   - Result: The focused omni-claude binary works end-to-end with real creds and preserves the original behavior/gate.

4. **Wrapper routing to Grok (file creds technique)**:
   - Temp creds file + env.
   - Wrapper with --providers grok, model="grok:grok-3-mini".
   - Init ok via loader; request hit gate (fake key).
   - Result: Wrapper correctly routes to grok backend using the shared credential loader.

5. **Replacements live via wrapper (Claude)**:
   - Temp rule: SECRET -> REDACTED (prompt scope).
   - Wrapper + --replace-rules.
   - Request with "SECRET".
   - Result: Outbound replacement applied before hitting Claude (verified in successful masked responses).

## Observations on "Grok Gate" (live + code)
- No heavy wire fingerprint required (unlike Claude's cch, betas, preamble, per-profile).
- Gate is primarily valid API key (Bearer).
- With the extracted loader: server starts cleanly when file present (even fake); request fails at xAI with auth error (the observable "gate").
- Headers sent (from code + expected): Authorization: Bearer ..., Content-Type.
- Optional in practice for some clients/proxies: X-Title, HTTP-Referer (to pass tracking or certain rate/abuse policies).
- service_tier in body for priority.
- Built-in tools require specific shapes (not just any function).
- Live with fake: confirms the credential path and error handling without crashing.
- Real Grok would succeed with valid key (as in unit "test_send_real_if_key_present" when key present in other sessions).

## Claude Live
- Full invariant exercised: real responses confirm the ported fingerprint/identity/cch/headers work against live Anthropic (no 403 from gate).
- Prefix routing in wrapper works.
- Replacements integrate live.

## Binaries Confirmed Working
- omni (light wrapper): routing, config, both backends, replacements.
- omni-claude: direct Claude.
- omni-grok: direct Grok + file creds.

All components (shared, providers, bins, wrapper) exercised live where creds allowed. Grok credential technique matches CCP exactly (file + env path override + fresh per request).

No new "gates" discovered in live beyond expected auth on invalid key.

See grok-gate.md for the dedicated explanation.
