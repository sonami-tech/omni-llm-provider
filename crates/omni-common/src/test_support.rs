//! Test-only helpers shared by the binary crates' subprocess integration tests.
//!
//! NOT used in production code paths. Lives here (rather than duplicated in each
//! binary's `mod tests`) so the build-and-locate logic has a single source of
//! truth across `omni`, `omni-grok`, and `omni-claude`.

use std::path::PathBuf;
use std::process::Command;

/// Build the named workspace binary on demand and return the path to the freshly
/// built executable.
///
/// WHY build on demand: `cargo test -p <bin>` compiles only that crate's unit-test
/// harness, not the standalone binary, and unit tests (unlike integration tests)
/// receive no `CARGO_BIN_EXE_*` env var. A subprocess test that just guessed
/// `target/debug/<bin>` would either find nothing (ENOENT) or, worse, silently run
/// a *stale* binary from a prior build after a handler/route change.
///
/// WHY parse `--message-format=json` instead of hardcoding `target/debug/<bin>`:
/// the hardcoded guess is wrong whenever `CARGO_TARGET_DIR` is set (common in CI),
/// so the test would spawn a missing or stale binary. Cargo reports the real
/// executable path in its `compiler-artifact` messages, so we ask cargo where it
/// put the binary rather than guess.
///
/// PROFILE: this always builds the default (dev) profile. It does NOT mirror a
/// `cargo test --release` parent, because a unit test in a bin crate has no
/// reliable way to learn the parent's profile. That is acceptable: the default
/// `cargo test` is dev, and the release/integration-test path is already covered
/// by the `CARGO_BIN_EXE_<pkg>` env var that cargo injects with the correct path
/// (call sites check it first). If a release-profile subprocess test is ever
/// needed, pass the profile explicitly rather than relying on this helper.
///
/// The build runs at most once per call; call sites cache the result in a
/// `OnceLock` so it runs once per test-binary process. Cargo's own artifact lock
/// makes any concurrent invocations safe regardless.
pub fn build_workspace_bin(package: &str) -> PathBuf {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let output = Command::new(cargo)
        .args(["build", "--message-format=json", "-p", package])
        .output()
        .unwrap_or_else(|e| panic!("invoke cargo build for {package}: {e}"));
    assert!(
        output.status.success(),
        "cargo build -p {package} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Scan compiler-artifact messages for the bin executable produced for `package`.
    // Require BOTH target.kind == ["bin"] AND target.name == package so we never
    // pick up a lib, build script, or example artifact; take the last match so a
    // re-link reports the current path.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut found: Option<PathBuf> = None;
    for line in stdout.lines() {
        let msg: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if msg.get("reason").and_then(|r| r.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let target = match msg.get("target") {
            Some(t) => t,
            None => continue,
        };
        let name_matches = target.get("name").and_then(|n| n.as_str()) == Some(package);
        let is_bin = target
            .get("kind")
            .and_then(|k| k.as_array())
            .is_some_and(|kinds| kinds.iter().any(|k| k.as_str() == Some("bin")));
        if !(name_matches && is_bin) {
            continue;
        }
        if let Some(exe) = msg.get("executable").and_then(|e| e.as_str()) {
            found = Some(PathBuf::from(exe));
        }
    }

    found.unwrap_or_else(|| {
        panic!("cargo build -p {package} produced no bin executable artifact for target '{package}'")
    })
}
