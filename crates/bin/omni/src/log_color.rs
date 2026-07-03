//! Color formatter for `tracing-subscriber` field rendering.
//!
//! Wraps `tracing_subscriber::fmt::format::DefaultFields` so timestamps,
//! level coloring, span scope, and event layout still come from upstream.
//! We only intercept per-field value rendering so identifier values
//! (`request_id`, `session_id`) get stable per-value colors, the closed set
//! of provider names (`claude`/`grok`/`codex`) gets fixed assigned colors,
//! and known state-change values (`finish_reason="tool_calls"` etc.) get a
//! fixed color cue. Field keys are left uncolored -- only values are tinted.
//!
//! Color is applied based on a runtime `ColorMode` resolved from
//! `OMNI_LOG_COLOR` and `NO_COLOR` plus stderr TTY detection. The same
//! formatter runs in containers and CI; it just emits no escape codes
//! when the output is not a TTY.
//!
//! Ported from the legacy `claude-code-provider` `log_color.rs`. Changes for
//! the multi-provider world: the `pid` field is gone (no subprocess model), and
//! `provider` is colored from a fixed three-entry table rather than the hashed
//! palette because it is a closed set that is more useful at a glance as a
//! consistent per-backend color.

use std::fmt::{self, Write as _};
use std::io::IsTerminal;

use nu_ansi_term::{Color, Style};
use tracing::field::{Field, Visit};
use tracing_subscriber::field::{MakeVisitor, VisitFmt, VisitOutput};
use tracing_subscriber::fmt::format::Writer;

/// Per-id stable palette. 256-color ANSI foreground codes chosen for
/// visibility on both light and dark terminals and reasonable separation
/// under deuteranopia. With 12 entries we saw frequent visual collisions
/// in normal load (a handful of sessions per minute landing on the same
/// hue), so we widen to 24 distinct mid-luminance shades. Avoid the very
/// dark (0-17) and very light (190+, 230+) ranges that disappear into
/// most terminal backgrounds, and avoid pure red/green-only contrast.
const PALETTE: &[u8] = &[
    27,  // blue
    33,  // cyan-blue
    39,  // deep cyan
    45,  // teal
    51,  // bright cyan
    57,  // indigo
    63,  // periwinkle
    75,  // sky blue
    81,  // aqua
    87,  // pale cyan
    93,  // violet
    105, // lavender
    111, // soft blue
    123, // pale aqua
    129, // purple
    135, // mauve
    141, // light purple
    147, // light lavender
    161, // magenta
    166, // orange
    172, // amber
    178, // yellow-brown
    70,  // green
    77,  // mid green
];

/// Resolved at startup. `Off` disables all color emission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColorMode {
    Off,
    On,
}

impl ColorMode {
    /// Resolve color policy from the process environment + stderr TTY detection.
    /// Thin wrapper over the pure [`ColorMode::resolve`] so the precedence logic
    /// is unit-testable without mutating global env (which would race the
    /// parallel test harness).
    pub fn from_env() -> Self {
        // Read NO_COLOR via var_os so a present-but-non-UTF8 value still forces
        // off (the de-facto convention keys on presence + non-emptiness, not on
        // the value being valid UTF-8). Map it to a stable non-empty sentinel for
        // the pure resolver, which only checks emptiness.
        let no_color = std::env::var_os("NO_COLOR").map(|v| {
            if v.is_empty() {
                String::new()
            } else {
                "1".to_string()
            }
        });
        let omni_log_color = std::env::var("OMNI_LOG_COLOR").unwrap_or_default();
        Self::resolve(no_color.as_deref(), &omni_log_color, || {
            std::io::stderr().is_terminal()
        })
    }

    /// Pure resolver. Precedence: `NO_COLOR` (any non-empty value) forces off --
    /// the de-facto convention. Then `OMNI_LOG_COLOR=always|never|auto` (case
    /// insensitive). Then `auto`: on iff `is_tty()` (only consulted in the auto
    /// case, so callers need not evaluate it eagerly).
    fn resolve(
        no_color: Option<&str>,
        omni_log_color: &str,
        is_tty: impl FnOnce() -> bool,
    ) -> Self {
        if no_color.is_some_and(|v| !v.is_empty()) {
            return ColorMode::Off;
        }
        match omni_log_color.to_ascii_lowercase().as_str() {
            "always" => ColorMode::On,
            "never" | "off" | "no" | "false" | "0" => ColorMode::Off,
            _ => {
                if is_tty() {
                    ColorMode::On
                } else {
                    ColorMode::Off
                }
            }
        }
    }
}

/// Field formatter that colors recognized identifier and state-change values.
#[derive(Clone, Copy)]
pub struct ColorFields {
    mode: ColorMode,
}

impl ColorFields {
    pub fn new(mode: ColorMode) -> Self {
        Self { mode }
    }
}

// We rely on tracing-subscriber's blanket impl `FormatFields for M where
// M: MakeOutput + ...` instead of implementing FormatFields directly. Wiring:
// MakeVisitor + ColorVisitor: VisitOutput<fmt::Result> + VisitFmt -> blanket
// MakeOutput -> blanket FormatFields. This avoids drift with upstream's
// timestamp/level/span-context layout while still letting us color values.
impl<'a> MakeVisitor<Writer<'a>> for ColorFields {
    type Visitor = ColorVisitor<'a>;

    fn make_visitor(&self, target: Writer<'a>) -> Self::Visitor {
        ColorVisitor::new(target, self.mode)
    }
}

pub struct ColorVisitor<'a> {
    writer: Writer<'a>,
    mode: ColorMode,
    first: bool,
    result: fmt::Result,
}

impl<'a> ColorVisitor<'a> {
    fn new(writer: Writer<'a>, mode: ColorMode) -> Self {
        Self {
            writer,
            mode,
            first: true,
            result: Ok(()),
        }
    }

    fn write_separator(&mut self) -> fmt::Result {
        if self.first {
            self.first = false;
            Ok(())
        } else {
            write!(self.writer, " ")
        }
    }

    /// Render a single field=value pair, applying value coloring when this
    /// field name is one we care about.
    ///
    /// EVERY value is ANSI-sanitized before emission, to prevent terminal-escape
    /// injection from any value carrying attacker- or upstream-controlled text
    /// (e.g. a provider echoing raw SSE bytes). Note this is deliberately
    /// STRONGER than upstream `DefaultVisitor`, which only wraps the `message`
    /// field and `record_error` output in an `EscapeGuard` and leaves ordinary
    /// field values as bare `{:?}`. `{:?}` on a `&str` already escapes control
    /// bytes, but a `%`/Display-sigil value (e.g. `warn!(error = %e)`) forwards
    /// its Display output verbatim, so per-field sanitization is the only defense
    /// there -- do not "restore parity" by dropping it. Our own color escapes are
    /// added AFTER sanitization, so they are never stripped.
    fn write_field(&mut self, name: &str, value: &dyn fmt::Debug) -> fmt::Result {
        // Skip fields that are `log`-crate metadata, matching upstream
        // DefaultVisitor. Skip BEFORE the separator so they consume no spacing.
        if is_log_metadata_field(name) {
            return Ok(());
        }

        self.write_separator()?;

        // Format the value once, then sanitize control sequences out of it.
        let mut raw = String::new();
        write!(&mut raw, "{:?}", value)?;
        let safe = sanitize_ansi(&raw);

        // The "message" field of an event is rendered without a key prefix
        // by the default formatter. Keep that contract.
        if name == "message" {
            return write!(self.writer, "{}", safe);
        }

        write!(self.writer, "{}=", name)?;

        // Non-color path: emit the sanitized value verbatim.
        if matches!(self.mode, ColorMode::Off) {
            return write!(self.writer, "{}", safe);
        }

        // Color path: style the already-sanitized value. style_for keys on the
        // sanitized text (unchanged for the identifiers/providers we color).
        match style_for(name, &safe) {
            Some(s) => write!(self.writer, "{}", s.paint(safe.as_str())),
            None => write!(self.writer, "{}", safe),
        }
    }
}

/// Whether a field is `log`-crate bridge metadata that upstream `DefaultVisitor`
/// suppresses. `tracing_subscriber::fmt().init()` installs a `LogTracer` (the
/// `tracing-log` default feature), so any `log`-facade record from a dependency
/// is bridged into `tracing` carrying `log.target` / `log.module_path` /
/// `log.file` / `log.line` fields. Rendering them would be noisy and would
/// diverge from upstream output, so they are dropped.
fn is_log_metadata_field(name: &str) -> bool {
    name.starts_with("log.")
}

/// Escape every control character an attacker could use for terminal injection
/// or log-line forging, rendering them as visible `\xNN` / `\u{..}` / `\r` /
/// `\n` sequences. This is deliberately STRONGER than `tracing_subscriber`'s
/// internal `EscapeGuard` (which escapes only ESC/BEL/BS/FF/DEL + the C1 range
/// and leaves the rest, including `\r`/`\n`/other C0, raw).
///
/// Policy: escape ALL C0 controls (`0x00`-`0x1f`) EXCEPT horizontal tab, escape
/// DEL (`0x7f`), and escape the C1 range (`0x80`-`0x9f`). Escaping the whole C0
/// block -- not just the handful that introduce escape sequences -- is defense
/// in depth: e.g. `\x0e`/`\x0f` (SO/SI) can switch a terminal's character set,
/// VT (`\x0b`) moves the cursor, and NUL can truncate downstream tooling. `\t`
/// is the sole exception: it is horizontal, common in legitimate values, and
/// cannot rewrite or forge output. `\r` and `\n` get readable `\r`/`\n`
/// spellings rather than `\x0d`/`\x0a`. No log call in this codebase emits
/// intentional multi-line values, so escaping `\n` costs nothing here.
fn sanitize_ansi(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\t' => out.push(ch), // sole C0 kept raw (horizontal, harmless)
            '\r' => out.push_str("\\r"),
            '\n' => out.push_str("\\n"),
            // All other C0 controls (0x00-0x1f) and DEL (0x7f) as \xNN.
            ch if (ch as u32) < 0x20 || (ch as u32) == 0x7f => {
                let _ = write!(out, "\\x{:02x}", ch as u32);
            }
            // C1 controls (0x80-0x9f) as \u{..}.
            ch if (0x80..=0x9f).contains(&(ch as u32)) => {
                let _ = write!(out, "\\u{{{:x}}}", ch as u32);
            }
            _ => out.push(ch),
        }
    }
    out
}

// `ColorVisitor` implements only `record_debug`. Upstream `DefaultVisitor` also
// overrides `record_str` and `record_error`. This is handled/safe as follows:
//   - The security-relevant behavior (ANSI sanitization) is applied uniformly in
//     `write_field` to EVERY value, so no path can leak a raw escape.
//   - `record_str` falls through to the default trait method, which is
//     `self.record_debug(field, &value)` (tracing-core) -- identical to what
//     upstream's `record_str` does for non-message fields. The message field is
//     never recorded via `record_str` (the macros always wrap it in
//     `format_args!`, hitting `record_debug`), so no divergence there either.
//   - `log.*` field skipping IS reproduced (in `write_field`): `fmt().init()`
//     installs a `LogTracer`, so those fields really do occur.
//   - `record_error` is NOT overridden. It is only invoked for a field recorded
//     as `&dyn std::error::Error`; every error we log uses the `%`/Display sigil
//     (a formatted value, not a `dyn Error`), so the upstream `record_error`
//     branch is never reached here. The default `record_error` routes through
//     `record_debug` (and thus sanitization). Raw-ident (`r#`) log fields
//     likewise do not occur in this codebase.
// If a `&dyn Error` field or a raw-ident log field is introduced, mirror the
// corresponding upstream branch here.
impl Visit for ColorVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        if self.result.is_err() {
            return;
        }
        self.result = self.write_field(field.name(), value);
    }
}

impl VisitOutput<fmt::Result> for ColorVisitor<'_> {
    fn finish(self) -> fmt::Result {
        self.result
    }
}

impl VisitFmt for ColorVisitor<'_> {
    fn writer(&mut self) -> &mut dyn fmt::Write {
        &mut self.writer
    }
}

/// Decide how to color a value based on its field name and (for known
/// state-change fields) the rendered value text.
fn style_for(field: &str, debug_text: &str) -> Option<Style> {
    match field {
        "request_id" | "session_id" => Some(palette_style(debug_text)),
        // Closed set of backends: a fixed color per provider reads more clearly
        // at a glance than a hash would, and stays stable across the whole set.
        "provider" => provider_style(debug_text.trim_matches('"')),
        "finish_reason" => match debug_text.trim_matches('"') {
            "tool_calls" => Some(Style::new().fg(Color::Fixed(178)).bold()),
            "stop" => Some(Style::new().fg(Color::Fixed(70))),
            _ => None,
        },
        _ => None,
    }
}

/// Fixed, distinct color per known provider. Unknown provider names are left
/// uncolored so a typo or a new backend is visually obvious rather than
/// silently sharing a hue.
fn provider_style(provider: &str) -> Option<Style> {
    let code = match provider {
        "claude" => 39, // deep cyan
        "grok" => 208,  // orange
        "codex" => 141, // light purple
        _ => return None,
    };
    Some(Style::new().fg(Color::Fixed(code)))
}

/// Stable per-value color from the curated palette.
fn palette_style(value: &str) -> Style {
    let h = fnv1a_64(value.as_bytes());
    let idx = (h as usize) % PALETTE.len();
    Style::new().fg(Color::Fixed(PALETTE[idx]))
}

/// FNV-1a 64-bit (canonical offset basis `0xcbf29ce484222325`, prime
/// `0x100000001b3`). Kept local to keep the log layer self-contained; the color
/// palette only needs a stable, well-distributed hash of the value bytes, so the
/// exact algorithm is not shared with any other module.
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

    #[test]
    fn palette_indices_in_range() {
        // Sanity: hashing many strings always lands inside palette.
        for s in ["a", "abc", "e9beeb2f", "h:af20f253c3a065c9", ""] {
            let style = palette_style(s);
            let _ = style; // just exercise; correctness is non-empty palette
        }
        assert!(!PALETTE.is_empty());
    }

    #[test]
    fn same_value_same_color() {
        assert_eq!(palette_style("abc"), palette_style("abc"));
    }

    #[test]
    fn finish_reason_styles() {
        assert!(style_for("finish_reason", "\"tool_calls\"").is_some());
        assert!(style_for("finish_reason", "\"stop\"").is_some());
        assert!(style_for("finish_reason", "\"unknown\"").is_none());
    }

    #[test]
    fn provider_colors_are_fixed_distinct_and_closed() {
        // Each known provider gets a color, and they differ from each other.
        let claude = style_for("provider", "\"claude\"").expect("claude colored");
        let grok = style_for("provider", "\"grok\"").expect("grok colored");
        let codex = style_for("provider", "\"codex\"").expect("codex colored");
        assert_ne!(claude, grok);
        assert_ne!(grok, codex);
        assert_ne!(claude, codex);
        // Fixed, not hashed: a provider must not pick up the palette color its
        // name would hash to.
        assert_ne!(Some(claude), Some(palette_style("claude")));
        // Unknown backend stays uncolored so it stands out.
        assert!(style_for("provider", "\"gemini\"").is_none());
    }

    #[test]
    fn log_bridge_fields_are_skipped() {
        // WHY: fmt().init() installs a LogTracer, so `log`-crate records arrive
        // as tracing events carrying log.target/log.module_path/log.file/log.line.
        // Upstream DefaultVisitor drops these; the color formatter must match, or
        // every bridged `log` record renders with extra noise fields.
        assert!(is_log_metadata_field("log.target"));
        assert!(is_log_metadata_field("log.module_path"));
        assert!(is_log_metadata_field("log.file"));
        assert!(is_log_metadata_field("log.line"));
        // Real event/span fields must NOT be skipped.
        assert!(!is_log_metadata_field("message"));
        assert!(!is_log_metadata_field("request_id"));
        assert!(!is_log_metadata_field("provider"));
        assert!(!is_log_metadata_field("login")); // starts with "log" but not "log."
    }

    #[test]
    fn unknown_fields_uncolored() {
        assert!(style_for("model", "\"opus\"").is_none());
        assert!(style_for("duration_ms", "1234").is_none());
        // pid was dropped in the multi-provider port; it must not be colored.
        assert!(style_for("pid", "1234").is_none());
    }

    // A MakeWriter backed by a shared buffer, so a scoped subscriber's output
    // can be inspected. Mirrors tracing-test's MockWriter; kept local so this
    // behavioral check does not depend on that crate's internals.
    #[derive(Clone)]
    struct BufWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for BufWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl tracing_subscriber::fmt::MakeWriter<'_> for BufWriter {
        type Writer = BufWriter;
        fn make_writer(&self) -> Self::Writer {
            self.clone()
        }
    }

    /// Render one event carrying colorable fields through a real subscriber that
    /// uses ColorFields, and return the raw captured bytes.
    fn render_line(mode: ColorMode) -> String {
        use tracing::info;
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(matches!(mode, ColorMode::On))
            .fmt_fields(ColorFields::new(mode))
            .with_writer(BufWriter(buf.clone()))
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            info!(session_id = "sess-xyz", provider = "grok", "hello");
        });
        String::from_utf8(buf.lock().unwrap().clone()).unwrap()
    }

    #[test]
    fn on_mode_emits_ansi_off_mode_is_plain() {
        // WHY: the acceptance criterion is behavioral -- the formatter must emit
        // escape codes when on and none when off, so a piped/redirected stream
        // (ColorMode::Off) stays clean. Unit-testing style_for alone would not
        // catch a subscriber-wiring regression.
        let colored = render_line(ColorMode::On);
        assert!(
            colored.contains('\u{1b}'),
            "On mode must emit ANSI escape codes, got: {colored:?}"
        );
        // The values we tint must be present regardless of coloring.
        assert!(colored.contains("sess-xyz"));
        assert!(colored.contains("grok"));

        let plain = render_line(ColorMode::Off);
        assert!(
            !plain.contains('\u{1b}'),
            "Off mode must emit no escape codes, got: {plain:?}"
        );
        // Off mode keeps upstream DefaultFields rendering verbatim, including the
        // quotes Debug adds around &str values.
        assert!(plain.contains("session_id=\"sess-xyz\""), "got: {plain:?}");
        assert!(plain.contains("provider=\"grok\""), "got: {plain:?}");
    }

    /// Render an event that carries BOTH a colorable field (so the formatter
    /// emits its own legitimate color escapes in On mode) AND attacker-controlled
    /// text in a non-colored field + the message, using a spread of injection
    /// primitives (CSI, OSC, DCS, bare C1, BEL). Returns the captured bytes.
    fn render_with_escape(mode: ColorMode) -> String {
        use tracing::info;
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(matches!(mode, ColorMode::On))
            .fmt_fields(ColorFields::new(mode))
            .with_writer(BufWriter(buf.clone()))
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            // Injection primitives a hostile upstream might smuggle: CSI color,
            // OSC title-set (ESC ] ... BEL), DCS (ESC P), a bare C1 CSI (0x9b),
            // and a raw BEL. `session_id` is colorable, so On mode DOES emit real
            // color codes -- the test must tolerate those but reject these.
            let evil = "x\x1b[31mCSI\x1b]0;title\x07\x1bPdcs\u{9b}C1\x07y";
            info!(
                session_id = "sess-xyz",
                data = evil,
                "upstream said: {evil}"
            );
        });
        String::from_utf8(buf.lock().unwrap().clone()).unwrap()
    }

    /// Strip the color escape sequences THIS formatter legitimately emits
    /// (`ESC [ ... m` SGR sequences) so the test can then assert that no OTHER
    /// raw control bytes -- i.e. anything smuggled through a logged value --
    /// survived. Removing only well-formed SGR (`m`-terminated CSI) is safe: the
    /// formatter never emits any other escape kind, so anything left is injected.
    fn strip_formatter_sgr(s: &str) -> String {
        let bytes = s.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut i = 0;
        while i < bytes.len() {
            // A legitimate SGR: ESC '[' <params> 'm'. Only strip that exact shape.
            if bytes[i] == 0x1b && i + 1 < bytes.len() && bytes[i + 1] == b'[' {
                let mut j = i + 2;
                while j < bytes.len() && (bytes[j] == b';' || bytes[j].is_ascii_digit()) {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'm' {
                    i = j + 1; // drop the whole SGR sequence
                    continue;
                }
            }
            out.push(bytes[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    #[test]
    fn ansi_escapes_in_values_are_neutralized_in_both_modes() {
        // WHY: values we log can carry attacker/upstream-controlled text (e.g.
        // provider-claude logs raw SSE bytes on a parse failure). Raw control
        // sequences in such a value would let the source drive the operator's
        // terminal (cursor moves, title/clipboard via OSC, clear-screen). Upstream
        // DefaultFields sanitizes via EscapeGuard; our formatter must match. After
        // removing the formatter's OWN legitimate SGR color codes, NO raw ESC/BEL
        // /C1 control byte may remain -- every such byte from a value must have
        // been rewritten to visible text (e.g. "\x1b", "\u{9b}").
        for mode in [ColorMode::Off, ColorMode::On] {
            let out = render_with_escape(mode);
            // Sanitization must have produced the escaped textual forms.
            assert!(
                out.contains("\\x1b") && out.contains("\\x07") && out.contains("\\u{9b}"),
                "{mode:?}: expected escaped control text missing, got: {out:?}"
            );
            // After stripping the formatter's own SGR color codes, nothing that
            // could drive a terminal may remain: no ESC (0x1b), no BEL (0x07), no
            // C1 CSI (0x9b).
            let residual = strip_formatter_sgr(&out);
            for bad in ['\u{1b}', '\u{07}', '\u{9b}'] {
                assert!(
                    !residual.contains(bad),
                    "{mode:?}: raw control {:#x} from an injected value survived: {residual:?}",
                    bad as u32
                );
            }
        }
        // Off mode emits no color at all, so the raw output itself must already be
        // free of every raw control byte (no stripping needed).
        let off = render_with_escape(ColorMode::Off);
        for bad in ['\u{1b}', '\u{07}', '\u{9b}'] {
            assert!(
                !off.contains(bad),
                "Off mode must contain no raw control byte {:#x}, got: {off:?}",
                bad as u32
            );
        }
    }

    #[test]
    fn display_sigil_value_is_sanitized() {
        // WHY: this is the path where `sanitize_ansi` is the SOLE defense. A
        // `&str` field goes through `{:?}`, which already escapes control bytes on
        // its own -- so the other ANSI test would pass even if sanitization were a
        // no-op for `&str`. But a `%`/Display-sigil field (used throughout the
        // codebase, e.g. `warn!(error = %e, ...)`) records a value whose Debug
        // forwards to Display verbatim, so a raw ESC in the Display output reaches
        // the writer unescaped UNLESS we sanitize. This locks in that behavior.
        use tracing::info;
        struct Evil;
        impl std::fmt::Display for Evil {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                // Raw CSI, a carriage return, and a newline, all emitted verbatim
                // by Display. `\r` overwrites the line; `\n` would forge a whole
                // fake log line ("FORGED ERROR ...") if not escaped.
                write!(f, "boom\x1b[2Jclear\rSPOOF\nFORGED ERROR line")
            }
        }
        for mode in [ColorMode::Off, ColorMode::On] {
            let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
            let subscriber = tracing_subscriber::fmt()
                .with_ansi(matches!(mode, ColorMode::On))
                .fmt_fields(ColorFields::new(mode))
                .with_writer(BufWriter(buf.clone()))
                .finish();
            tracing::subscriber::with_default(subscriber, || {
                info!(error = %Evil, "display-sigil");
            });
            let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
            // ESC, CR, and LF from the value must all be escaped to visible text.
            assert!(
                out.contains("\\x1b"),
                "{mode:?}: ESC not sanitized: {out:?}"
            );
            assert!(out.contains("\\r"), "{mode:?}: CR not sanitized: {out:?}");
            assert!(out.contains("\\n"), "{mode:?}: LF not sanitized: {out:?}");
            // The whole event must be ONE physical line: the only real newline is
            // the formatter's trailing terminator, so the value's `\n` did not
            // forge a second line. (A forged line would make this 2 lines of
            // content.)
            let content_lines: Vec<&str> = out.trim_end_matches('\n').split('\n').collect();
            assert_eq!(
                content_lines.len(),
                1,
                "{mode:?}: value's newline forged an extra log line: {out:?}"
            );
            // "FORGED ERROR line" must appear only inline (never at line start).
            assert!(
                !out.contains("\nFORGED"),
                "{mode:?}: forged line reached output: {out:?}"
            );
            let residual = strip_formatter_sgr(out.trim_end_matches('\n'));
            // No raw ESC / CR / LF from the value may survive in the content.
            for (bad, name) in [('\u{1b}', "ESC"), ('\r', "CR"), ('\n', "LF")] {
                assert!(
                    !residual.contains(bad),
                    "{mode:?}: raw {name} from a %-Display value survived: {residual:?}"
                );
            }
        }
    }

    #[test]
    fn sanitize_escapes_all_c0_controls_except_tab() {
        // WHY: the hardening goal is "no control byte from a value can drive the
        // terminal". Escaping only the escape-sequence introducers would leave
        // SO/SI (charset switch), VT (cursor down), NUL (truncation), etc. raw.
        // Policy: every C0 (0x00-0x1f) + DEL + C1 is escaped, tab alone stays raw.
        for b in 0u32..=0x1f {
            let ch = char::from_u32(b).unwrap();
            let out = sanitize_ansi(&ch.to_string());
            if ch == '\t' {
                assert_eq!(out, "\t", "tab must stay raw");
            } else {
                assert!(!out.contains(ch), "C0 {b:#04x} left raw in {out:?}");
                // Readable spellings for CR/LF, \xNN for the rest.
                let expected = match ch {
                    '\r' => "\\r".to_string(),
                    '\n' => "\\n".to_string(),
                    _ => format!("\\x{b:02x}"),
                };
                assert_eq!(out, expected, "C0 {b:#04x} wrong escaping");
            }
        }
        // DEL and a representative C1.
        assert_eq!(sanitize_ansi("\x7f"), "\\x7f");
        assert_eq!(sanitize_ansi("\u{9b}"), "\\u{9b}");
        // Ordinary printable text and tab are untouched.
        assert_eq!(sanitize_ansi("hello\tworld"), "hello\tworld");
    }

    #[test]
    fn color_mode_resolution_precedence() {
        // Pure resolver: no process-env mutation, so this is safe under the
        // parallel test harness (the previous env-mutating version raced other
        // tests and relied on a false single-threaded assumption).
        // NO_COLOR wins over everything, including OMNI_LOG_COLOR=always.
        assert_eq!(
            ColorMode::resolve(Some("1"), "always", || true),
            ColorMode::Off
        );
        // Empty NO_COLOR does NOT force off (de-facto convention: any non-empty).
        assert_eq!(
            ColorMode::resolve(Some(""), "always", || false),
            ColorMode::On
        );
        // OMNI_LOG_COLOR explicit values, case-insensitive.
        assert_eq!(ColorMode::resolve(None, "always", || false), ColorMode::On);
        assert_eq!(ColorMode::resolve(None, "ALWAYS", || false), ColorMode::On);
        assert_eq!(ColorMode::resolve(None, "never", || true), ColorMode::Off);
        assert_eq!(ColorMode::resolve(None, "off", || true), ColorMode::Off);
        // auto (unset/unknown) follows the TTY probe, which is only consulted here.
        assert_eq!(ColorMode::resolve(None, "", || true), ColorMode::On);
        assert_eq!(ColorMode::resolve(None, "", || false), ColorMode::Off);
        assert_eq!(ColorMode::resolve(None, "auto", || true), ColorMode::On);
    }
}
