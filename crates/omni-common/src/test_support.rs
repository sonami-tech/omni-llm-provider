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
/// the hardcoded guess is wrong whenever `CARGO_TARGET_DIR` is set (common in CI)
/// or a non-`dev` profile is used (`--release`). Cargo reports the real executable
/// path in its `compiler-artifact` messages, so we ask cargo where it put the
/// binary rather than guess.
///
/// The build runs at most once per `package` per test-binary process (cached via
/// the returned path through a per-call `OnceLock` keyed implicitly by the caller's
/// own `Once`). Callers typically wrap this so it runs once; cargo's own artifact
/// lock makes concurrent invocations safe regardless.
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

    // Scan compiler-artifact messages for the executable produced for `package`.
    // We take the last matching executable so a re-link reports the current path.
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
        // Match the target name to the package's binary (target name == bin name).
        let is_bin = msg
            .get("target")
            .and_then(|t| t.get("name"))
            .and_then(|n| n.as_str())
            == Some(package);
        if !is_bin {
            continue;
        }
        if let Some(exe) = msg.get("executable").and_then(|e| e.as_str()) {
            found = Some(PathBuf::from(exe));
        }
    }

    found.unwrap_or_else(|| {
        panic!("cargo build -p {package} produced no executable artifact for target '{package}'")
    })
}
