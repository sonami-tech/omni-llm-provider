//! Shared provider version + catalog abstraction.
//!
//! Every provider that tracks an upstream client (Claude Code, grok-shell, Codex
//! CLI) exposes one or more [`ProviderVersion`]s. A version pairs a client
//! version string with two model catalogs:
//!
//! - **conservative**: the models the real client advertises / offers. In
//!   conservative mode the provider also targets the client's exact wire
//!   protocol (byte-for-byte parity).
//! - **extended**: any model probed and verified to work on the most optimal
//!   surface available to us (a superset of, or different from, conservative).
//!   Extended is the default mode.
//!
//! The catalog *data* is defined inside each provider (so the provider owns its
//! own truth); core only defines the shape and the selection logic. Core obtains
//! a provider's versions by querying the provider, never by hardcoding models.

use std::fmt;

/// Which catalog a request resolves against.
///
/// Extended is the default: it uses whatever working surface is most optimal for
/// us. Conservative restricts to the models the real client offers and implies
/// exact client-protocol parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CatalogMode {
    /// Only models the real client advertises; exact client-protocol parity.
    Conservative,
    /// Any model verified to work on the most optimal surface (default).
    #[default]
    Extended,
}

impl CatalogMode {
    pub fn as_str(self) -> &'static str {
        match self {
            CatalogMode::Conservative => "conservative",
            CatalogMode::Extended => "extended",
        }
    }
}

impl fmt::Display for CatalogMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One model in a provider catalog: a real upstream id plus inbound-only aliases.
///
/// Aliases are accepted on the way in but are not part of the advertised model
/// surface (mirrors how the providers already treat aliases).
#[derive(Debug, Clone, Copy)]
pub struct CatalogModel {
    pub id: &'static str,
    pub aliases: &'static [&'static str],
}

impl CatalogModel {
    pub const fn new(id: &'static str, aliases: &'static [&'static str]) -> Self {
        Self { id, aliases }
    }

    /// True if `input` matches this model's id or any alias (case-sensitive, as
    /// upstream ids are).
    pub fn matches(&self, input: &str) -> bool {
        self.id == input || self.aliases.contains(&input)
    }
}

/// A single client version with its two catalogs.
///
/// `conservative` and `extended` are both real upstream id lists. `default_model`
/// is the id selected when no model is specified; it must appear in `extended`
/// (and normally in `conservative` too).
#[derive(Debug, Clone, Copy)]
pub struct ProviderVersion {
    /// The full client version string, e.g. "2.1.186", "0.2.60", "0.142.0". The
    /// complete string is significant and must match exactly for exact pins.
    pub version: &'static str,
    /// Models the real client advertises (conservative mode).
    pub conservative: &'static [CatalogModel],
    /// Models verified to work on the optimal surface (extended mode, default).
    pub extended: &'static [CatalogModel],
    /// Id chosen when the caller specifies no model.
    pub default_model: &'static str,
}

impl ProviderVersion {
    /// The catalog for a given mode.
    pub fn catalog(&self, mode: CatalogMode) -> &'static [CatalogModel] {
        match mode {
            CatalogMode::Conservative => self.conservative,
            CatalogMode::Extended => self.extended,
        }
    }

    /// Resolve `input` (id or alias) to a real upstream id within `mode`'s
    /// catalog. Returns `None` when the input matches nothing in that catalog.
    pub fn resolve_model(&self, input: &str, mode: CatalogMode) -> Option<&'static str> {
        self.catalog(mode)
            .iter()
            .find(|m| m.matches(input))
            .map(|m| m.id)
    }
}

/// How a caller selects which version to pin across the whole system.
///
/// `Latest` is the default (newest in each provider's catalog). The two
/// match-system variants differ only in strictness; `Exact` never does a fuzzy
/// match.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum VersionSelector {
    /// Newest version in the provider's catalog (default when no flag is given).
    #[default]
    Latest,
    /// Pin this exact version string. Must match a catalog version exactly or the
    /// resolution fails (no closest match).
    Exact(String),
    /// Match the installed-on-this-system client version, choosing the closest
    /// catalog version when there is no exact match.
    MatchSystem(String),
    /// Match the installed-on-this-system client version, requiring an exact
    /// catalog version; fail loudly if the catalog lacks it.
    MatchSystemExact(String),
}

/// Why a version selection failed (so callers can fail loudly with a clear msg).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionResolveError {
    /// An exact pin (`--client-version` or `--match-system-exact`) named a version
    /// the provider catalog does not contain.
    ExactNotFound {
        requested: String,
        available: Vec<String>,
    },
    /// The provider exposes no versions at all (cannot select).
    NoVersions,
}

impl fmt::Display for VersionResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VersionResolveError::ExactNotFound {
                requested,
                available,
            } => write!(
                f,
                "no exact match for version {requested:?}; available: [{}]",
                available.join(", ")
            ),
            VersionResolveError::NoVersions => f.write_str("provider exposes no versions"),
        }
    }
}

impl std::error::Error for VersionResolveError {}

/// Resolve a [`VersionSelector`] against a provider's version list.
///
/// `versions` is the provider's own catalog, newest-first (index 0 is newest).
/// Returns the chosen version or a [`VersionResolveError`] for the fail-loud
/// cases.
pub fn resolve_version<'v>(
    versions: &'v [ProviderVersion],
    selector: &VersionSelector,
) -> Result<&'v ProviderVersion, VersionResolveError> {
    let newest = versions.first().ok_or(VersionResolveError::NoVersions)?;
    match selector {
        VersionSelector::Latest => Ok(newest),
        VersionSelector::Exact(want) | VersionSelector::MatchSystemExact(want) => versions
            .iter()
            .find(|v| v.version == want)
            .ok_or_else(|| VersionResolveError::ExactNotFound {
                requested: want.clone(),
                available: versions.iter().map(|v| v.version.to_string()).collect(),
            }),
        VersionSelector::MatchSystem(want) => {
            // Exact first, else closest by version-component distance, else newest.
            if let Some(exact) = versions.iter().find(|v| v.version == want) {
                return Ok(exact);
            }
            // Guard against empty / non-numeric detection output (e.g. ""): it would
            // otherwise parse to [0] and "closest-match" against version 0, masking
            // a detection failure. Fall back to newest explicitly instead.
            if !has_numeric_component(want) {
                return Ok(newest);
            }
            Ok(closest_version(versions, want).unwrap_or(newest))
        }
    }
}

/// Pick the catalog version closest to `want`.
///
/// "Closest" is decided by [`version_distance`], which compares the per-component
/// absolute differences lexicographically from most-significant to least. This
/// guarantees a more-significant component always dominates a less-significant one
/// regardless of magnitude (no overflow-prone weighting). Returns `None` only for
/// an empty slice.
fn closest_version<'v>(versions: &'v [ProviderVersion], want: &str) -> Option<&'v ProviderVersion> {
    let target = parse_version(want);
    versions.iter().min_by(|a, b| {
        version_distance(&target, &parse_version(a.version))
            .cmp(&version_distance(&target, &parse_version(b.version)))
    })
}

fn parse_version(s: &str) -> Vec<u64> {
    s.split('.')
        .map(|part| part.trim().parse::<u64>().unwrap_or(0))
        .collect()
}

/// True if `s` has at least one dotted component that parses as a number. Used to
/// reject empty / fully non-numeric detection output before fuzzy matching.
fn has_numeric_component(s: &str) -> bool {
    s.split('.').any(|part| part.trim().parse::<u64>().is_ok())
}

/// Per-component absolute-difference vector between two dotted version vectors,
/// most-significant component first, so a lexicographic `Ord` on the result makes
/// a more-significant component dominate a less-significant one regardless of
/// magnitude. (Replaces an earlier base-1000 weighted sum that lost dominance once
/// a component reached 1000 - flagged by external review and covered by a test.)
fn version_distance(a: &[u64], b: &[u64]) -> Vec<u64> {
    let len = a.len().max(b.len());
    let mut diffs = Vec::with_capacity(len);
    for i in 0..len {
        let av = a.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        diffs.push(av.abs_diff(bv));
    }
    diffs
}

#[cfg(test)]
mod tests {
    use super::*;

    const M_A: &[CatalogModel] = &[CatalogModel::new("m-a", &["a"])];
    const M_AB: &[CatalogModel] = &[
        CatalogModel::new("m-a", &["a"]),
        CatalogModel::new("m-b", &["b", "bee"]),
    ];

    const V_NEW: ProviderVersion = ProviderVersion {
        version: "2.1.186",
        conservative: M_A,
        extended: M_AB,
        default_model: "m-a",
    };
    const V_OLD: ProviderVersion = ProviderVersion {
        version: "2.1.175",
        conservative: M_A,
        extended: M_A,
        default_model: "m-a",
    };
    const VERSIONS: &[ProviderVersion] = &[V_NEW, V_OLD];

    #[test]
    fn extended_is_default_mode() {
        assert_eq!(CatalogMode::default(), CatalogMode::Extended);
    }

    #[test]
    fn conservative_and_extended_select_different_catalogs() {
        // Extended sees both models; conservative only the advertised one. This is
        // the whole point of the split - a regression here would silently expose
        // or hide models from the wrong surface.
        assert_eq!(V_NEW.catalog(CatalogMode::Conservative).len(), 1);
        assert_eq!(V_NEW.catalog(CatalogMode::Extended).len(), 2);
        assert_eq!(
            V_NEW.resolve_model("bee", CatalogMode::Extended),
            Some("m-b")
        );
        assert_eq!(V_NEW.resolve_model("bee", CatalogMode::Conservative), None);
    }

    #[test]
    fn latest_picks_newest() {
        let v = resolve_version(VERSIONS, &VersionSelector::Latest).unwrap();
        assert_eq!(v.version, "2.1.186");
    }

    #[test]
    fn exact_pin_must_match_or_fail() {
        // Exact pin found.
        let v = resolve_version(VERSIONS, &VersionSelector::Exact("2.1.175".into())).unwrap();
        assert_eq!(v.version, "2.1.175");
        // Exact pin missing -> hard error, NOT a closest match. This is the
        // contract: --client-version is exact-or-fail.
        let err = resolve_version(VERSIONS, &VersionSelector::Exact("2.1.180".into())).unwrap_err();
        match err {
            VersionResolveError::ExactNotFound { requested, .. } => {
                assert_eq!(requested, "2.1.180")
            }
            other => panic!("expected ExactNotFound, got {other:?}"),
        }
    }

    #[test]
    fn match_system_exact_fails_loudly_when_absent() {
        let err = resolve_version(VERSIONS, &VersionSelector::MatchSystemExact("9.9.9".into()))
            .unwrap_err();
        assert!(matches!(err, VersionResolveError::ExactNotFound { .. }));
    }

    #[test]
    fn match_system_picks_closest_when_no_exact() {
        // 2.1.180 has no exact entry; closest is 2.1.186 (distance 6) over 2.1.175
        // (distance 5)... wait: |180-186|=6, |180-175|=5, so 2.1.175 is closer.
        let v = resolve_version(VERSIONS, &VersionSelector::MatchSystem("2.1.180".into())).unwrap();
        assert_eq!(v.version, "2.1.175");
        // 2.1.184 -> |184-186|=2 vs |184-175|=9 -> 2.1.186 wins.
        let v = resolve_version(VERSIONS, &VersionSelector::MatchSystem("2.1.184".into())).unwrap();
        assert_eq!(v.version, "2.1.186");
    }

    #[test]
    fn major_component_dominates_distance() {
        // A far-off patch on the same major/minor beats a near patch on a
        // different major: 2.1.999 should still be closer to 2.1.186 than 3.0.0.
        const V3: ProviderVersion = ProviderVersion {
            version: "3.0.0",
            conservative: M_A,
            extended: M_A,
            default_model: "m-a",
        };
        const VS: &[ProviderVersion] = &[V3, V_NEW];
        let v = resolve_version(VS, &VersionSelector::MatchSystem("2.1.999".into())).unwrap();
        assert_eq!(v.version, "2.1.186");
    }

    #[test]
    fn major_dominates_even_when_minor_component_exceeds_1000() {
        // Regression guard (external review): a more-significant component must
        // dominate regardless of magnitude. Target 2.0.0; candidate 2.5000.0 shares
        // the major and must win over 3.0.0, even though the minor gap (5000) is far
        // larger than the major gap (1). The old base-1000 weighting got this wrong.
        const V2_BIG: ProviderVersion = ProviderVersion {
            version: "2.5000.0",
            conservative: M_A,
            extended: M_A,
            default_model: "m-a",
        };
        const V3: ProviderVersion = ProviderVersion {
            version: "3.0.0",
            conservative: M_A,
            extended: M_A,
            default_model: "m-a",
        };
        const VS: &[ProviderVersion] = &[V3, V2_BIG];
        let v = resolve_version(VS, &VersionSelector::MatchSystem("2.0.0".into())).unwrap();
        assert_eq!(
            v.version, "2.5000.0",
            "same-major candidate must win over a different-major one"
        );
    }

    #[test]
    fn match_system_with_empty_or_nonnumeric_input_falls_back_to_newest() {
        // Regression guard (external review): empty / non-numeric detection output
        // must NOT silently fuzzy-match version 0; it falls back to newest.
        for bad in ["", "   ", "unknown", "not.a.version"] {
            let v = resolve_version(VERSIONS, &VersionSelector::MatchSystem(bad.into())).unwrap();
            assert_eq!(
                v.version, "2.1.186",
                "empty/non-numeric {bad:?} must fall back to newest, not match version 0"
            );
        }
    }
}
