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
    /// Resolve color policy from env + TTY detection.
    ///
    /// Precedence: `NO_COLOR` (any non-empty value) forces off -- this is the
    /// de-facto convention. Then `OMNI_LOG_COLOR=always|never|auto` (case
    /// insensitive). Then `auto`: on iff stderr is a TTY.
    pub fn from_env() -> Self {
        if std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()) {
            return ColorMode::Off;
        }
        match std::env::var("OMNI_LOG_COLOR")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "always" => ColorMode::On,
            "never" | "off" | "no" | "false" | "0" => ColorMode::Off,
            _ => {
                if std::io::stderr().is_terminal() {
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
    /// Field/message values are ANSI-sanitized before emission, matching
    /// upstream `DefaultFields` (which wraps values in an `EscapeGuard`). This
    /// prevents terminal-escape injection from any value we log that carries
    /// attacker- or upstream-controlled text (e.g. a provider echoing raw SSE
    /// bytes). Our own color escapes are added AFTER sanitization, so they are
    /// never stripped.
    fn write_field(&mut self, name: &str, value: &dyn fmt::Debug) -> fmt::Result {
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

/// Escape the control characters an attacker could use for terminal injection,
/// rendering them as visible `\xNN` / `\u{..}` sequences. Byte-for-byte matches
/// `tracing_subscriber`'s internal `EscapeGuard` (its `escape.rs`): C0 controls
/// used in escape sequences (ESC/BEL/BS/FF/DEL) plus the C1 range (0x80-0x9f).
/// Ordinary text (including normal whitespace like `\n`/`\t`) is untouched.
fn sanitize_ansi(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\x1b' => out.push_str("\\x1b"),
            '\x07' => out.push_str("\\x07"),
            '\x08' => out.push_str("\\x08"),
            '\x0c' => out.push_str("\\x0c"),
            '\x7f' => out.push_str("\\x7f"),
            ch if (ch as u32) >= 0x80 && (ch as u32) <= 0x9f => {
                let _ = write!(out, "\\u{{{:x}}}", ch as u32);
            }
            _ => out.push(ch),
        }
    }
    out
}

// `ColorVisitor` implements only `record_debug`. Upstream `DefaultVisitor` also
// special-cases `record_str` (message-without-quotes), `record_error`
// (`.sources=` chain), `log.*` field skipping, and `r#` raw-ident stripping. We
// deliberately do NOT reimplement those, and it is safe here:
//   - The security-relevant behavior (ANSI sanitization) is applied uniformly in
//     `write_field` to EVERY value, so no path can leak a raw escape.
//   - `record_str` for non-message fields falls through to the default trait
//     method, which calls our `record_debug` (`&value` -> Debug + sanitize),
//     matching upstream's non-message `record_debug(&value)`.
//   - `record_error`, `r#` fields, and `log.*` fields never occur in this
//     codebase: all `error=` logs use the `%`/Display sigil (never `&dyn Error`),
//     there are no raw-ident log fields, and `tracing-log`/`LogTracer` is not
//     installed (only the `env-filter` feature is enabled), so no `log.*` fields
//     are ever emitted. Reimplementing them would be dead code (see Rule 2).
// If any of those preconditions change (a `LogTracer`, a `&dyn Error` field, a
// raw-ident field), revisit this and mirror the corresponding upstream branch.
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

/// FNV-1a 64-bit, identical to `session::fnv1a_64`. Duplicated to avoid a
/// public re-export and to keep the log layer self-contained.
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

    /// Render an event whose message and a field value carry a raw ESC, and
    /// return the captured bytes.
    fn render_with_escape(mode: ColorMode) -> String {
        use tracing::info;
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(matches!(mode, ColorMode::On))
            .fmt_fields(ColorFields::new(mode))
            .with_writer(BufWriter(buf.clone()))
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            // A provider echoing upstream bytes could inject this into a log.
            let evil = "before\x1b[31mRED\x1b[0mafter";
            info!(data = evil, "upstream said: {evil}");
        });
        String::from_utf8(buf.lock().unwrap().clone()).unwrap()
    }

    #[test]
    fn ansi_escapes_in_values_are_neutralized_in_both_modes() {
        // WHY: values we log can carry attacker/upstream-controlled text (e.g.
        // provider-claude logs raw SSE bytes on a parse failure). A raw ESC in
        // such a value would let the source drive the operator's terminal
        // (cursor moves, color, clear-screen). Upstream DefaultFields sanitizes
        // these via EscapeGuard; our custom formatter must match. The raw ESC
        // byte (0x1b) must NEVER reach output as an in-value control; it must be
        // escaped to the visible text "\x1b".
        for mode in [ColorMode::Off, ColorMode::On] {
            let out = render_with_escape(mode);
            // The injected value's ESC must appear only in escaped textual form.
            assert!(
                out.contains("\\x1b"),
                "{mode:?}: sanitized \\x1b text missing, got: {out:?}"
            );
            // In Off mode there must be NO raw ESC byte at all. In On mode the
            // ONLY raw ESC bytes permitted are our own color codes wrapping the
            // colored id/provider fields -- never from the injected `data`/message
            // value. Assert the injected payload did not smuggle a raw sequence:
            // the literal "RED" must not be immediately preceded by a raw ESC-[.
            assert!(
                !out.contains("\x1b[31mRED"),
                "{mode:?}: raw injected ANSI reached output: {out:?}"
            );
            assert!(
                !out.contains("RED\x1b[0mafter"),
                "{mode:?}: raw injected ANSI reset reached output: {out:?}"
            );
        }
        // Off mode: the entire line is free of raw ESC bytes.
        let off = render_with_escape(ColorMode::Off);
        assert!(
            !off.contains('\u{1b}'),
            "Off mode must contain no raw ESC byte, got: {off:?}"
        );
    }

    #[test]
    fn no_color_env_disables() {
        // Save and restore env for hermetic test.
        let prev_no = std::env::var("NO_COLOR").ok();
        let prev_omni = std::env::var("OMNI_LOG_COLOR").ok();
        // SAFETY: tests run single-threaded by default per crate test harness;
        // these env reads happen before the formatter is initialized in main.
        unsafe {
            std::env::set_var("NO_COLOR", "1");
            std::env::set_var("OMNI_LOG_COLOR", "always");
        }
        assert_eq!(ColorMode::from_env(), ColorMode::Off);
        unsafe {
            match prev_no {
                Some(v) => std::env::set_var("NO_COLOR", v),
                None => std::env::remove_var("NO_COLOR"),
            }
            match prev_omni {
                Some(v) => std::env::set_var("OMNI_LOG_COLOR", v),
                None => std::env::remove_var("OMNI_LOG_COLOR"),
            }
        }
    }
}
