# Grok (xAI) Provider Requirements and Findings

## API Compatibility
- Base URL: https://api.x.ai/v1
- Primary endpoint for chat: POST /v1/chat/completions (OpenAI-compatible)
- Also supports /v1/responses for advanced agentic features (stateful, built-in tools)

## Authentication
- Header: `Authorization: Bearer $XAI_API_KEY`
- Key obtained from https://console.x.ai/
- No special OAuth subscription gate like Claude Code. Straightforward API key.

## Headers for "Gates" / Access
- Standard: Authorization + Content-Type: application/json
- For some client libraries or proxies (e.g., to pass usage tracking or avoid blocks in certain setups):
  - X-Title: (e.g., "my-app")
  - HTTP-Referer: (e.g., "https://example.com")
- These are not strictly required for direct API use but recommended for enterprise/priority access or to identify the client in logs/rate limits.
- service_tier: "default" | "priority" (passed in body for /chat/completions or responses)
- For built-in tools (web_search, x_search, code_execution): use specific tool objects in the request, or search_parameters.

## Interesting Findings
- No byte-exact fingerprint/cch/preamble like Claude. Much lighter "gate".
- Reasoning models (grok-4.3 etc.) support "reasoning_effort": "low"|"medium"|"high" (in body).
- Built-in server-side tools are a big differentiator; they execute on xAI side and return citations/usage.
- Responses API is preferred for multi-turn agent loops (previous_response_id).
- No special "Claude Code" style identity injection needed.

## Usage in Omni
- Provider-grok implements via reqwest to the above, mapping CanonicalRequest <-> xAI shapes.
- Replacements hooks applied (prompt/response scopes) as in common.
- Focus of current work: real calls + tools + reasoning.

See also: xAI official docs (quickstart, rest-api-reference), and the provider-grok source for exact header/body construction.
