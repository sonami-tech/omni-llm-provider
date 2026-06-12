//! Test-only helpers for subprocess integration tests.
//!
//! NOT used in production code paths. Lives here so binary integration tests
//! have one source of truth for build lookup and HTTP probing.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

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
        panic!(
            "cargo build -p {package} produced no bin executable artifact for target '{package}'"
        )
    })
}

/// True only when a developer explicitly opted into tests that call real
/// provider APIs. Credential presence alone is intentionally not enough.
pub fn live_tests_enabled() -> bool {
    std::env::var("OMNI_LIVE_TESTS")
        .map(|v| matches!(v.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

#[derive(Debug)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

pub fn http_get(url: impl AsRef<str>) -> HttpResponse {
    http_request("GET", url.as_ref(), &[], None).unwrap_or_else(|e| panic!("{e}"))
}

pub fn http_get_with_headers(url: impl AsRef<str>, headers: &[(&str, &str)]) -> HttpResponse {
    http_request("GET", url.as_ref(), headers, None).unwrap_or_else(|e| panic!("{e}"))
}

pub fn http_post_json(url: impl AsRef<str>, body: impl AsRef<str>) -> HttpResponse {
    http_request(
        "POST",
        url.as_ref(),
        &[("content-type", "application/json")],
        Some(body.as_ref()),
    )
    .unwrap_or_else(|e| panic!("{e}"))
}

pub fn http_post_json_with_headers(
    url: impl AsRef<str>,
    headers: &[(&str, &str)],
    body: impl AsRef<str>,
) -> HttpResponse {
    let mut merged = vec![("content-type", "application/json")];
    merged.extend_from_slice(headers);
    http_request("POST", url.as_ref(), &merged, Some(body.as_ref()))
        .unwrap_or_else(|e| panic!("{e}"))
}

fn http_request(
    method: &str,
    url: &str,
    headers: &[(&str, &str)],
    body: Option<&str>,
) -> Result<HttpResponse, String> {
    let parsed = parse_http_url(url)?;
    let mut stream = TcpStream::connect(&parsed.addr)
        .map_err(|e| format!("connect to {} for {method} {url}: {e}", parsed.addr))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| format!("set read timeout for {url}: {e}"))?;
    stream
        .set_write_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| format!("set write timeout for {url}: {e}"))?;

    let body = body.unwrap_or("");
    let mut req = format!(
        "{method} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nContent-Length: {}\r\n",
        parsed.path,
        parsed.host_header,
        body.len()
    );
    for (name, value) in headers {
        req.push_str(name);
        req.push_str(": ");
        req.push_str(value);
        req.push_str("\r\n");
    }
    req.push_str("\r\n");
    req.push_str(body);

    stream
        .write_all(req.as_bytes())
        .map_err(|e| format!("write request for {method} {url}: {e}"))?;

    let mut raw = Vec::new();
    stream
        .read_to_end(&mut raw)
        .map_err(|e| format!("read response for {method} {url}: {e}"))?;
    let split_at = find_bytes(&raw, b"\r\n\r\n")
        .ok_or_else(|| format!("HTTP response missing header/body separator: {raw:?}"))?;
    let head = String::from_utf8(raw[..split_at].to_vec())
        .map_err(|e| format!("HTTP response headers for {method} {url} are not utf8: {e}"))?;
    let body_bytes = &raw[split_at + 4..];
    let mut lines = head.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| format!("HTTP response missing status line: {raw:?}"))?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| format!("HTTP response status line malformed: {status_line:?}"))?
        .parse::<u16>()
        .map_err(|e| format!("parse HTTP status from {status_line:?}: {e}"))?;
    let is_chunked = lines.any(|line| {
        let lower = line.to_ascii_lowercase();
        lower.starts_with("transfer-encoding:") && lower.contains("chunked")
    });
    let body_bytes = if is_chunked {
        decode_chunked_body(body_bytes)?
    } else {
        body_bytes.to_vec()
    };
    let body = String::from_utf8(body_bytes)
        .map_err(|e| format!("response body for {method} {url} is not utf8: {e}"))?;
    Ok(HttpResponse { status, body })
}

struct ParsedHttpUrl {
    addr: String,
    host_header: String,
    path: String,
}

fn parse_http_url(url: &str) -> Result<ParsedHttpUrl, String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("test HTTP helper only supports http:// URLs: {url}"))?;
    let (host_port, path) = rest
        .split_once('/')
        .map(|(h, p)| (h, format!("/{p}")))
        .unwrap_or((rest, "/".to_string()));
    let addr = if host_port.contains(':') {
        host_port.to_string()
    } else {
        format!("{host_port}:80")
    };
    Ok(ParsedHttpUrl {
        addr,
        host_header: host_port.to_string(),
        path,
    })
}

fn decode_chunked_body(mut body: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let line_end = find_bytes(body, b"\r\n")
            .ok_or_else(|| format!("chunked body missing size line terminator: {body:?}"))?;
        let size_line = std::str::from_utf8(&body[..line_end])
            .map_err(|e| format!("chunk size line is not utf8: {e}"))?;
        let rest = &body[line_end + 2..];
        let size_hex = size_line.split(';').next().unwrap_or(size_line).trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|e| format!("parse chunk size {size_hex:?}: {e}"))?;
        if size == 0 {
            break;
        }
        if rest.len() < size + 2 {
            return Err(format!(
                "chunked body too short for chunk size {size}: remaining {} bytes",
                rest.len()
            ));
        }
        out.extend_from_slice(&rest[..size]);
        let tail = &rest[size..];
        if !tail.starts_with(b"\r\n") {
            return Err("chunk missing trailing CRLF".to_string());
        }
        body = &tail[2..];
    }
    Ok(out)
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Wait until a GET request returns the expected body with status 200.
pub fn wait_for_http_body(url: impl AsRef<str>, body: &str, timeout: Duration) -> bool {
    wait_for_http_body_with_headers(url, &[], body, timeout)
}

pub fn wait_for_http_body_with_headers(
    url: impl AsRef<str>,
    headers: &[(&str, &str)],
    body: &str,
    timeout: Duration,
) -> bool {
    let url = url.as_ref();
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Ok(resp) = http_request("GET", url, headers, None) {
            if resp.status == 200 && resp.body.trim() == body {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(120));
    }
    false
}

/// RAII wrapper that stops only the subprocess this test started.
pub struct ChildGuard {
    child: Option<Child>,
}

impl ChildGuard {
    pub fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub fn parse_json(body: &str) -> serde_json::Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("parse json body failed: {e}\n{body}"))
}

pub fn model_ids(v: &serde_json::Value) -> Vec<String> {
    v["data"]
        .as_array()
        .unwrap_or_else(|| panic!("models response missing data array: {v}"))
        .iter()
        .filter_map(|m| m["id"].as_str().map(str::to_string))
        .collect()
}

pub fn object_fields(v: &serde_json::Value) -> &serde_json::Map<String, serde_json::Value> {
    v.as_object()
        .unwrap_or_else(|| panic!("expected json object, got: {v}"))
}

pub fn as_object_map(v: &serde_json::Value) -> HashMap<String, serde_json::Value> {
    object_fields(v)
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}
