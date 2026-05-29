//! Session-id derivation for log correlation.
//!
//! The OpenAI chat-completions protocol is stateless: every turn is an
//! independent HTTP request with the full message history. To group log
//! lines from a single multi-turn agent session (typically a tool-call
//! loop) we derive a `session_id` per request from, in order:
//!
//! 1. `x-session-id` header              (origin: `x:`)
//! 2. OpenAI `user` body field           (origin: `u:`)
//! 3. Hash of a stable session anchor    (origin: `h:`)
//! 4. `-`                                (no fingerprint available)
//!
//! The hash anchor is the first user message text (plus the full system
//! prompt, the model, and the API key id as a tenant discriminator). This
//! anchor stays constant across every turn of one conversation: the agent
//! loop appends tool-result and assistant messages but the leading system
//! and first user message do not change.
//!
//! Hashing is done with FNV-1a 64-bit, which is stable across process
//! restarts (unlike `std::collections::hash_map::DefaultHasher`, which
//! reseeds per process). Output is rendered as 16 hex chars.

use crate::translate::request::{ChatCompletionRequest, ChatMessage, extract_text};

const MAX_EXPLICIT_LEN: usize = 128;

/// Resolve a session id for this request. Returns a string ready to log.
pub fn resolve_session_id(
	header_value: Option<&str>,
	request: &ChatCompletionRequest,
	api_key_id: Option<&str>,
) -> String {
	if let Some(v) = header_value.and_then(sanitize_explicit) {
		return format!("x:{}", v);
	}
	if let Some(v) = request.user.as_deref().and_then(sanitize_explicit) {
		return format!("u:{}", v);
	}
	if !has_anchor_material(request, api_key_id) {
		return "-".to_string();
	}
	let anchor = build_hash_anchor(request, api_key_id);
	format!("h:{:016x}", fnv1a_64(anchor.as_bytes()))
}

/// Whether we have anything stable enough to derive a hash anchor from. If a
/// request has no api key id, no system prompt, and no first user message,
/// the hash would just fingerprint the model — every such request would
/// share an id, which is misleading. Return `-` instead.
fn has_anchor_material(request: &ChatCompletionRequest, api_key_id: Option<&str>) -> bool {
	if api_key_id.is_some() {
		return true;
	}
	for msg in &request.messages {
		if msg.role == "system" && !extract_text(&msg.content).is_empty() {
			return true;
		}
	}
	!first_user_text(&request.messages).is_empty()
}

/// Sanitize an operator-supplied session identifier for safe log embedding.
/// Replaces whitespace and control characters with `_` (rather than deleting,
/// which could collapse distinct values like `"alice bob"` and `"alicebob"`),
/// and caps the result to MAX_EXPLICIT_LEN *bytes* (on a char boundary) so a
/// multi-byte id cannot bloat log lines or the derived filename. Returns
/// `None` for empty input.
fn sanitize_explicit(value: &str) -> Option<String> {
	let trimmed = value.trim();
	if trimmed.is_empty() {
		return None;
	}
	let mut cleaned = String::with_capacity(trimmed.len().min(MAX_EXPLICIT_LEN));
	for c in trimmed.chars() {
		let mapped = if c.is_control() || c.is_whitespace() { '_' } else { c };
		// Stop before exceeding the byte budget; never split a char.
		if cleaned.len() + mapped.len_utf8() > MAX_EXPLICIT_LEN {
			break;
		}
		cleaned.push(mapped);
	}
	if cleaned.is_empty() { None } else { Some(cleaned) }
}

/// Build the deterministic hash input. We include:
/// - api key id (tenant separator; identical prompts from different keys
///   should not collide in logs),
/// - the model string,
/// - all leading system messages concatenated,
/// - the first user message text.
fn build_hash_anchor(request: &ChatCompletionRequest, api_key_id: Option<&str>) -> String {
	let mut s = String::with_capacity(256);
	s.push_str(api_key_id.unwrap_or("-"));
	s.push('\x1f');
	s.push_str(&request.model);
	s.push('\x1f');
	for msg in &request.messages {
		if msg.role == "system" {
			s.push_str(&extract_text(&msg.content));
			s.push('\x1f');
		}
	}
	s.push_str(&first_user_text(&request.messages));
	s
}

fn first_user_text(messages: &[ChatMessage]) -> String {
	for msg in messages {
		// The session anchor only needs the first true user message.
		if msg.role == "user" {
			let text = extract_text(&msg.content);
			if !text.is_empty() {
				return text;
			}
		}
	}
	String::new()
}

/// FNV-1a 64-bit. Stable across process restarts and dependency-free.
fn fnv1a_64(bytes: &[u8]) -> u64 {
	const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
	const PRIME: u64 = 0x100_0000_01b3;
	let mut h = OFFSET;
	for &b in bytes {
		h ^= b as u64;
		h = h.wrapping_mul(PRIME);
	}
	h
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::translate::request::{ChatMessage, MessageContent};

	fn msg(role: &str, content: &str) -> ChatMessage {
		ChatMessage {
			role: role.to_string(),
			content: Some(MessageContent::Text(content.to_string())),
			..Default::default()
		}
	}

	fn req(messages: Vec<ChatMessage>) -> ChatCompletionRequest {
		ChatCompletionRequest {
			model: "opus".to_string(),
			messages,
			..Default::default()
		}
	}

	#[test]
	fn header_takes_precedence_over_user_field() {
		let mut r = req(vec![msg("user", "hi")]);
		r.user = Some("body-user".into());
		let id = resolve_session_id(Some("hdr-session"), &r, Some("k1"));
		assert_eq!(id, "x:hdr-session");
	}

	#[test]
	fn user_field_used_when_no_header() {
		let mut r = req(vec![msg("user", "hi")]);
		r.user = Some("body-user".into());
		let id = resolve_session_id(None, &r, Some("k1"));
		assert_eq!(id, "u:body-user");
	}

	#[test]
	fn hash_fallback_when_no_explicit_id() {
		let r = req(vec![msg("user", "hi")]);
		let id = resolve_session_id(None, &r, Some("k1"));
		assert!(id.starts_with("h:"));
		assert_eq!(id.len(), 2 + 16);
	}

	#[test]
	fn hash_is_stable_across_turns_of_same_conversation() {
		// First turn: system + first user only.
		let turn1 = req(vec![
			msg("system", "be helpful"),
			msg("user", "what time is it?"),
		]);
		// Second turn: agent loop appended assistant+tool messages and a new
		// user follow-up, but the system + first user are unchanged.
		let turn2 = req(vec![
			msg("system", "be helpful"),
			msg("user", "what time is it?"),
			msg("assistant", "<tool_call>...</tool_call>"),
			msg("tool", "12:34"),
			msg("user", "thanks"),
		]);
		let a = resolve_session_id(None, &turn1, Some("k1"));
		let b = resolve_session_id(None, &turn2, Some("k1"));
		assert_eq!(a, b, "hash anchor must be stable across multi-turn loops");
	}

	#[test]
	fn hash_differs_across_tenants() {
		let r = req(vec![msg("user", "hello")]);
		let a = resolve_session_id(None, &r, Some("k1"));
		let b = resolve_session_id(None, &r, Some("k2"));
		assert_ne!(a, b);
	}

	#[test]
	fn hash_differs_across_models() {
		let mut r = req(vec![msg("user", "hello")]);
		let a = resolve_session_id(None, &r, Some("k1"));
		r.model = "haiku".to_string();
		let b = resolve_session_id(None, &r, Some("k1"));
		assert_ne!(a, b);
	}

	#[test]
	fn empty_or_whitespace_explicit_falls_back() {
		let r = req(vec![msg("user", "hi")]);
		let id = resolve_session_id(Some("   "), &r, Some("k1"));
		assert!(id.starts_with("h:"));
	}

	#[test]
	fn explicit_value_is_length_capped() {
		let long = "a".repeat(500);
		let r = req(vec![msg("user", "hi")]);
		let id = resolve_session_id(Some(&long), &r, None);
		assert_eq!(id.len(), 2 + MAX_EXPLICIT_LEN);
	}

	#[test]
	fn whitespace_and_controls_replaced_with_underscore() {
		// Distinct values must remain distinct after sanitization.
		let r = req(vec![msg("user", "hi")]);
		let id = resolve_session_id(Some("alice bob"), &r, None);
		assert_eq!(id, "x:alice_bob");
		let id2 = resolve_session_id(Some("alicebob"), &r, None);
		assert_ne!(id, id2);
		let id3 = resolve_session_id(Some("ab\nc\x00d"), &r, None);
		assert_eq!(id3, "x:ab_c_d");
	}

	#[test]
	fn no_anchor_material_falls_back_to_dash() {
		// No api key, no system, no user message text.
		let r = req(vec![msg("assistant", "hello")]);
		let id = resolve_session_id(None, &r, None);
		assert_eq!(id, "-");
	}

	#[test]
	fn api_key_alone_is_enough_anchor() {
		let r = req(vec![msg("assistant", "hello")]);
		let id = resolve_session_id(None, &r, Some("k1"));
		assert!(id.starts_with("h:"));
	}

	#[test]
	fn fnv1a_known_vector() {
		// Empty string FNV-1a 64 is the offset basis.
		assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
		// "a" -> 0xaf63dc4c8601ec8c
		assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
	}
}
