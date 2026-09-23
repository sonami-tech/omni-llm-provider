//! Claude-specific model catalog, alias resolution, and wire defaults.
//! Ported/adapted from reference-src-claude/models.rs .
//! This is Claude-only; the catalog and resolution rules are part of the
//! fingerprint invariant (exact models the pinned Claude Code version accepts).
//! Nothing here is exposed to omni-core canonical types.

use serde::Serialize;

#[derive(Debug)]
pub struct ModelDef {
    pub canonical: &'static str,
    pub cli_name: &'static str,
    pub aliases: &'static [&'static str],
    pub context_window: u64,
    pub max_tokens: u64,
}

/// Active Claude Code model catalog (pin 2.1.280).
/// Fable alias is `claude-fable-5-1`. Opus is `claude-opus-5-5`, sonnet is
/// `claude-sonnet-5`, and haiku is dated. Explicit `claude-fable-5` stays
/// pass-through plus a wire/beta override, not a second advertised row.
pub static MODEL_CATALOG: &[ModelDef] = &[
    ModelDef {
        canonical: "claude-fable-5-1",
        cli_name: "fable",
        aliases: &["fable"],
        context_window: 1_000_000,
        max_tokens: 64_000,
    },
    ModelDef {
        canonical: "claude-opus-5-5",
        cli_name: "opus",
        aliases: &["opus"],
        context_window: 1_000_000,
        max_tokens: 128_000,
    },
    ModelDef {
        canonical: "claude-sonnet-5",
        cli_name: "sonnet",
        aliases: &["sonnet"],
        context_window: 1_000_000,
        max_tokens: 64_000,
    },
    ModelDef {
        canonical: "claude-haiku-4-5-20251001",
        cli_name: "haiku",
        aliases: &["haiku"],
        context_window: 200_000,
        max_tokens: 64_000,
    },
];

/// Resolve an input model string within one Claude Code profile catalog.
///
/// Resolution is exact-only: exact canonical, then exact alias. An unknown
/// model returns `None` so callers forward it verbatim (pass-through) rather
/// than rewriting it to a family canonical or a profile default. This is the
/// deliberate replacement for the former substring/default-fallback behavior,
/// which silently remapped ids like `claude-sonnet-5` onto another model.
pub fn resolve_model_in_catalog(
    input: &str,
    models: &'static [ModelDef],
) -> Option<&'static ModelDef> {
    for m in models {
        if m.canonical == input {
            return Some(m);
        }
    }

    for m in models {
        for alias in m.aliases {
            if *alias == input {
                return Some(m);
            }
        }
    }

    None
}

/// Return the model list for GET /v1/models using one profile catalog.
pub fn models_list_from_catalog(models: &'static [ModelDef]) -> Vec<ModelInfo> {
    models
        .iter()
        .map(|m| ModelInfo {
            id: m.canonical.to_string(),
            object: "model",
            created: 0,
            owned_by: "anthropic",
            context_window: m.context_window,
            max_tokens: m.max_tokens,
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn catalog_contains_unique_names(models: &'static [ModelDef]) -> bool {
    for (idx, model) in models.iter().enumerate() {
        if model.canonical.is_empty() || model.cli_name.is_empty() {
            return false;
        }
        for other in &models[idx + 1..] {
            if model.canonical == other.canonical || model.cli_name == other.cli_name {
                return false;
            }
            if other.aliases.contains(&model.canonical) || model.aliases.contains(&other.canonical)
            {
                return false;
            }
            if other.aliases.contains(&model.cli_name) || model.aliases.contains(&other.cli_name) {
                return false;
            }
            if model
                .aliases
                .iter()
                .any(|alias| other.aliases.contains(alias))
            {
                return false;
            }
        }
        for (alias_idx, alias) in model.aliases.iter().enumerate() {
            if alias.is_empty() || *alias == model.canonical {
                return false;
            }
            if model.aliases[alias_idx + 1..].contains(alias) {
                return false;
            }
        }
    }
    true
}

#[derive(Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    pub context_window: u64,
    pub max_tokens: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::FingerprintProfile;
    use crate::fingerprint::default_profile;
    fn profile() -> &'static FingerprintProfile {
        default_profile()
    }

    #[test]
    fn resolve_canonical_names() {
        assert_eq!(
            profile()
                .resolve_model("claude-opus-5-5")
                .unwrap()
                .canonical,
            "claude-opus-5-5"
        );
        assert_eq!(
            profile()
                .resolve_model("claude-sonnet-5")
                .unwrap()
                .canonical,
            "claude-sonnet-5"
        );
        // Only the exact catalog canonical resolves; the non-dated short form
        // `claude-haiku-4-5` is NOT a catalog entry (the canonical is dated) and
        // is no longer rewritten by a substring match - it passes through raw.
        assert!(profile().resolve_model("claude-haiku-4-5").is_none());
        // Retired opus id is not in the active catalog.
        assert!(profile().resolve_model("claude-opus-4-8").is_none());
    }

    #[test]
    fn resolve_short_aliases() {
        assert_eq!(
            profile().resolve_model("opus").unwrap().canonical,
            "claude-opus-5-5"
        );
        assert_eq!(
            profile().resolve_model("sonnet").unwrap().canonical,
            "claude-sonnet-5"
        );
        assert_eq!(
            profile().resolve_model("haiku").unwrap().canonical,
            "claude-haiku-4-5-20251001"
        );
        assert_eq!(
            profile().resolve_model("fable").unwrap().canonical,
            "claude-fable-5-1"
        );
    }

    #[test]
    fn resolve_claude_prefix_longforms_pass_through() {
        // WHY: `claude-opus`/`claude-sonnet`/`claude-haiku` are NOT catalog
        // aliases (MODEL_CATALOG has only the short forms). They resolved
        // only via the deleted substring matcher. Under pure pass-through they
        // return None and forward raw (owner-accepted: Anthropic 400s them).
        assert!(profile().resolve_model("claude-opus").is_none());
        assert!(profile().resolve_model("claude-sonnet").is_none());
        assert!(profile().resolve_model("claude-haiku").is_none());
    }

    #[test]
    fn resolve_date_suffixed_passes_through() {
        // WHY: dated variants that are not an exact catalog canonical were only
        // matched by substring; that matcher is deleted, so they return None and
        // forward raw. The one exact canonical still resolves.
        assert!(
            profile()
                .resolve_model("claude-opus-4-8-20260101")
                .is_none()
        );
        assert!(
            profile()
                .resolve_model("claude-opus-4-6-20260101")
                .is_none()
        );
        assert!(
            profile()
                .resolve_model("claude-sonnet-4-6-20260101")
                .is_none()
        );
        assert_eq!(
            profile()
                .resolve_model("claude-haiku-4-5-20251001")
                .unwrap()
                .canonical,
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn old_opus_canonical_passes_through() {
        // WHY: the retired dated id `claude-opus-4-6` is not in this profile's
        // catalog; it only resolved via substring. Now it passes through raw.
        assert!(profile().resolve_model("claude-opus-4-6").is_none());
    }

    #[test]
    fn resolve_unknown_returns_none() {
        // WHY: an unknown model no longer falls back to a profile default; it
        // returns None so callers forward it verbatim (pass-through). This is the
        // fix for the silent-remap bug (`claude-sonnet-5` -> another model).
        assert!(profile().resolve_model("gpt-4").is_none());
        assert!(profile().resolve_model("unknown").is_none());
        assert!(profile().resolve_model("").is_none());
    }

    #[test]
    fn models_list_returns_default_catalog() {
        let list = profile().models_list();
        assert_eq!(list.len(), 4);
        assert_eq!(list[0].id, "claude-fable-5-1");
        assert_eq!(list[1].id, "claude-opus-5-5");
        assert_eq!(list[2].id, "claude-sonnet-5");
        assert_eq!(list[3].id, "claude-haiku-4-5-20251001");
        assert_eq!(list[0].context_window, 1_000_000);
        assert_eq!(list[3].max_tokens, 64_000);
    }

    #[test]
    fn profile_catalog_names_are_unique() {
        assert!(catalog_contains_unique_names(profile().models));
    }

    #[test]
    fn resolve_canonical_exact() {
        assert_eq!(
            profile().resolve_model("claude-opus-5-5").unwrap().cli_name,
            "opus"
        );
        assert_eq!(
            profile().resolve_model("claude-sonnet-5").unwrap().cli_name,
            "sonnet"
        );
    }

    #[test]
    fn resolve_via_cli_name_direct() {
        // cli_name is the "spoken" alias in Claude Code UX.
        assert_eq!(profile().resolve_model("opus").unwrap().cli_name, "opus");
        assert_eq!(profile().resolve_model("haiku").unwrap().cli_name, "haiku");
    }

    #[test]
    fn resolve_substring_family_variants_pass_through() {
        // WHY: substring family matching is deleted. Ids that merely CONTAIN a
        // cli_name (`haiku-20251001`, `something-haiku-dated`) are no longer
        // rewritten to a family canonical; they return None and forward raw.
        assert!(profile().resolve_model("haiku-20251001").is_none());
        assert!(profile().resolve_model("something-haiku-dated").is_none());
    }

    #[test]
    fn family_longforms_pass_through_on_active_pin() {
        // WHY: bare family long-forms and retired dated ids are not catalog
        // entries; they must pass through raw so Anthropic rejects them instead
        // of a silent remap. Short aliases still resolve on the active pin.
        let p = profile();
        for longform in [
            "claude-opus",
            "claude-sonnet",
            "claude-haiku",
            "claude-opus-4-6",
        ] {
            assert!(
                p.resolve_model(longform).is_none(),
                "{longform} must pass through raw on active pin"
            );
        }
        assert!(p.resolve_model("opus").is_some());
        assert!(p.resolve_model("sonnet").is_some());
        assert!(p.resolve_model("haiku").is_some());
    }
}
