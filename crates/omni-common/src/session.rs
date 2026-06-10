// Simplified session derivation for omni (original had OAI request type import).
// In full design, frontends provide a small "SessionContext" (user id, header) to this.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub fn resolve_session_id(
    header: Option<&str>,
    user: Option<&str>,
    api_key_id: Option<&str>,
) -> String {
    if let Some(h) = header
        && !h.is_empty()
    {
        return h.to_string();
    }
    if let Some(u) = user
        && !u.is_empty()
    {
        return format!("user:{}", u);
    }
    if let Some(k) = api_key_id {
        return format!("key:{}", k);
    }
    // fallback stable hash
    let mut hasher = DefaultHasher::new();
    "default".hash(&mut hasher);
    format!("anon:{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefers_header() {
        assert_eq!(resolve_session_id(Some("hdr"), Some("u"), Some("k")), "hdr");
    }
    #[test]
    fn falls_back_to_user() {
        assert_eq!(resolve_session_id(None, Some("u"), None), "user:u");
    }

    #[test]
    fn falls_back_to_key() {
        assert_eq!(resolve_session_id(None, None, Some("k123")), "key:k123");
    }

    #[test]
    fn header_takes_precedence_over_all() {
        assert_eq!(resolve_session_id(Some("hdr"), Some("u"), Some("k")), "hdr");
    }

    #[test]
    fn empty_header_falls_to_user() {
        assert_eq!(resolve_session_id(Some(""), Some("u"), None), "user:u");
    }

    #[test]
    fn anon_fallback_is_stable() {
        let a = resolve_session_id(None, None, None);
        let b = resolve_session_id(None, None, None);
        assert_eq!(a, b);
        assert!(a.starts_with("anon:"));
    }

    // more derivation: user empty string falls through to key (like empty header)
    #[test]
    fn empty_user_falls_to_key() {
        assert_eq!(resolve_session_id(None, Some(""), Some("k9")), "key:k9");
    }

    // key takes precedence over anon, header over user even if user present
    #[test]
    fn full_precedence_header_user_key() {
        assert_eq!(resolve_session_id(Some("h"), Some("u"), Some("k")), "h");
        assert_eq!(resolve_session_id(None, Some("u"), Some("k")), "user:u");
        assert_eq!(resolve_session_id(None, None, Some("k")), "key:k");
    }

    // non-empty header (incl whitespace-only) is used verbatim; only exactly empty falls through.
    // (simplified omni session impl; full CCP sanitizes+trims explicit values.)
    #[test]
    fn whitespace_header_is_used_as_is() {
        assert_eq!(resolve_session_id(Some("   "), Some("u"), None), "   ");
    }

    // more precedence: empty user falls to key (user "" is empty so skips to key).
    #[test]
    fn empty_user_falls_to_key_more() {
        assert_eq!(resolve_session_id(None, Some(""), Some("k9")), "key:k9");
        assert_eq!(resolve_session_id(Some(""), Some(""), Some("kX")), "key:kX");
    }

    // full header/user/key precedence matrix.
    #[test]
    fn full_header_user_key_precedence() {
        assert_eq!(resolve_session_id(Some("h"), Some("u"), Some("k")), "h");
        assert_eq!(resolve_session_id(None, Some("u"), Some("k")), "user:u");
        assert_eq!(resolve_session_id(None, None, Some("k")), "key:k");
        assert_eq!(resolve_session_id(Some("hdr"), None, None), "hdr");
    }

    // whitespace header used as-is (non-empty); empty user + key.
    #[test]
    fn whitespace_header_used_as_is_and_empty_user_key() {
        assert_eq!(
            resolve_session_id(Some("   "), Some("user"), Some("k")),
            "   "
        );
        assert_eq!(
            resolve_session_id(Some(""), Some("   "), Some("k1")),
            "user:   "
        );
        assert_eq!(resolve_session_id(Some(""), Some(""), Some("k1")), "key:k1");
    }

    // empty header falls to user (or key).
    #[test]
    fn empty_header_falls_to_user_or_key() {
        assert_eq!(resolve_session_id(Some(""), Some("u1"), None), "user:u1");
        assert_eq!(resolve_session_id(Some(""), None, Some("k1")), "key:k1");
    }
}
