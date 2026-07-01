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

pub static CATALOG_CC_2_1_142: &[ModelDef] = &[
    ModelDef {
        canonical: "claude-opus-4-7",
        cli_name: "opus",
        aliases: &["opus", "claude-opus", "claude-opus-4-6"],
        context_window: 1_000_000,
        max_tokens: 128_000,
    },
    ModelDef {
        canonical: "claude-sonnet-4-6",
        cli_name: "sonnet",
        aliases: &["sonnet", "claude-sonnet"],
        context_window: 1_000_000,
        max_tokens: 64_000,
    },
    ModelDef {
        canonical: "claude-haiku-4-5",
        cli_name: "haiku",
        aliases: &["haiku", "claude-haiku"],
        context_window: 200_000,
        max_tokens: 64_000,
    },
];

pub static CATALOG_CC_2_1_150: &[ModelDef] = &[
    ModelDef {
        canonical: "claude-opus-4-7",
        cli_name: "opus",
        aliases: &["opus", "claude-opus", "claude-opus-4-6"],
        context_window: 1_000_000,
        max_tokens: 128_000,
    },
    ModelDef {
        canonical: "claude-sonnet-4-6",
        cli_name: "sonnet",
        aliases: &["sonnet", "claude-sonnet"],
        context_window: 1_000_000,
        max_tokens: 64_000,
    },
    ModelDef {
        canonical: "claude-haiku-4-5-20251001",
        cli_name: "haiku",
        aliases: &["haiku", "claude-haiku", "claude-haiku-4-5"],
        context_window: 200_000,
        max_tokens: 64_000,
    },
];

pub static CATALOG_CC_2_1_154: &[ModelDef] = &[
    ModelDef {
        canonical: "claude-opus-4-8",
        cli_name: "opus",
        aliases: &["opus"],
        context_window: 1_000_000,
        max_tokens: 128_000,
    },
    ModelDef {
        canonical: "claude-sonnet-4-6",
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

/// 2.1.158 catalog: identical to 2.1.154 (no model-list GET was present in the
/// 2026-05-30 capture, so no confirmed renames or window changes). Bodies
/// confirmed claude-opus-4-8, claude-sonnet-4-6, and claude-haiku-4-5 (non-dated)
/// are accepted by the CLI. Haiku canonical kept as dated per 154 for alias
/// resolution consistency; non-dated form is handled via overrides elsewhere.
pub static CATALOG_CC_2_1_158: &[ModelDef] = &[
    ModelDef {
        canonical: "claude-opus-4-8",
        cli_name: "opus",
        aliases: &["opus"],
        context_window: 1_000_000,
        max_tokens: 128_000,
    },
    ModelDef {
        canonical: "claude-sonnet-4-6",
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

/// 2.1.175 catalog captured 2026-06-12 from the installed CLI plus clean
/// fake-server probes. Fable is newly surfaced by the CLI; its advertised max
/// tokens are kept at the confirmed 64k wire value.
pub static CATALOG_CC_2_1_175: &[ModelDef] = &[
    ModelDef {
        canonical: "claude-fable-5",
        cli_name: "fable",
        aliases: &["fable"],
        context_window: 1_000_000,
        max_tokens: 64_000,
    },
    ModelDef {
        canonical: "claude-opus-4-8",
        cli_name: "opus",
        aliases: &["opus"],
        context_window: 1_000_000,
        max_tokens: 64_000,
    },
    ModelDef {
        canonical: "claude-sonnet-4-6",
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

/// Validate the reasoning_effort field.
/// Returns the effort string to pass to --effort, or None if the flag should be omitted.
/// (Adapted: returns Result<Option<&'static str>, String> to avoid pulling AppError here.)
pub fn validate_effort(effort: Option<&str>) -> Result<Option<&'static str>, String> {
    match effort {
        None => Ok(None),
        Some("none") => Ok(None),
        Some("low") => Ok(Some("low")),
        Some("medium") => Ok(Some("medium")),
        Some("high") => Ok(Some("high")),
        Some("max") => Ok(Some("max")),
        Some(other) => Err(format!(
            "Invalid reasoning_effort: '{}'. Valid values: none, low, medium, high, max",
            other
        )),
    }
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
                .resolve_model("claude-opus-4-8")
                .unwrap()
                .canonical,
            "claude-opus-4-8"
        );
        assert_eq!(
            profile()
                .resolve_model("claude-sonnet-4-6")
                .unwrap()
                .canonical,
            "claude-sonnet-4-6"
        );
        // Only the exact catalog canonical resolves; the non-dated short form
        // `claude-haiku-4-5` is NOT a catalog entry (the canonical is dated) and
        // is no longer rewritten by a substring match - it passes through raw.
        assert!(profile().resolve_model("claude-haiku-4-5").is_none());
    }

    #[test]
    fn resolve_short_aliases() {
        assert_eq!(
            profile().resolve_model("opus").unwrap().canonical,
            "claude-opus-4-8"
        );
        assert_eq!(
            profile().resolve_model("sonnet").unwrap().canonical,
            "claude-sonnet-4-6"
        );
        assert_eq!(
            profile().resolve_model("haiku").unwrap().canonical,
            "claude-haiku-4-5-20251001"
        );
    }

    #[test]
    fn resolve_claude_prefix_longforms_pass_through() {
        // WHY: `claude-opus`/`claude-sonnet`/`claude-haiku` are NOT catalog
        // aliases (CATALOG_CC_2_1_175 has only the short forms). They resolved
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
    fn validate_effort_valid_values() {
        assert_eq!(validate_effort(None).unwrap(), None);
        assert_eq!(validate_effort(Some("none")).unwrap(), None);
        assert_eq!(validate_effort(Some("low")).unwrap(), Some("low"));
        assert_eq!(validate_effort(Some("medium")).unwrap(), Some("medium"));
        assert_eq!(validate_effort(Some("high")).unwrap(), Some("high"));
        assert_eq!(validate_effort(Some("max")).unwrap(), Some("max"));
    }

    #[test]
    fn validate_effort_invalid() {
        assert!(validate_effort(Some("extreme")).is_err());
        assert!(validate_effort(Some("")).is_err());
    }

    #[test]
    fn models_list_returns_default_catalog() {
        let list = profile().models_list();
        assert_eq!(list.len(), 4);
        assert_eq!(list[0].id, "claude-fable-5");
        assert_eq!(list[1].id, "claude-opus-4-8");
        assert_eq!(list[2].id, "claude-sonnet-4-6");
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
            profile().resolve_model("claude-opus-4-8").unwrap().cli_name,
            "opus"
        );
        assert_eq!(
            profile()
                .resolve_model("claude-sonnet-4-6")
                .unwrap()
                .cli_name,
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
    fn resolve_unknown_returns_none_for_all_profiles() {
        // WHY: no profile falls back to a default model any more; an unknown id
        // returns None everywhere so the caller forwards it raw.
        for p in &[
            crate::fingerprint::PROFILE_CLAUDE_2_1_165_SDK_CLI,
            crate::fingerprint::PROFILE_CLAUDE_2_1_162_SDK_CLI,
            crate::fingerprint::PROFILE_CLAUDE_2_1_158_SDK_CLI,
            crate::fingerprint::PROFILE_CLAUDE_2_1_154_SDK_CLI,
        ] {
            assert!(
                p.resolve_model("nonexistent-xyz").is_none(),
                "unknown model must not resolve for {}",
                p.name
            );
        }
    }
}
