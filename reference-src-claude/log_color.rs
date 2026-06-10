//! Color formatter for `tracing-subscriber` field rendering.
//!
//! Wraps `tracing_subscriber::fmt::format::DefaultFields` so timestamps,
//! level coloring, span scope, and event layout still come from upstream.
//! We only intercept per-field value rendering so identifier values
//! (`request_id`, `session_id`) get stable per-value colors and known
//! state-change values (`finish_reason="tool_calls"` etc.) get a fixed
//! color cue. Field keys are left uncolored — only values are tinted.
//!
//! Color is applied based on a runtime `ColorMode` resolved from
//! `CCP_LOG_COLOR` and `NO_COLOR` plus stderr TTY detection. The same
//! formatter runs in containers and CI; it just emits no escape codes
//! when the output is not a TTY.

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
/// dark (0–17) and very light (190+, 230+) ranges that disappear into
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
	/// Precedence: `NO_COLOR` (any non-empty value) forces off — this is the
	/// de-facto convention. Then `CCP_LOG_COLOR=always|never|auto` (case
	/// insensitive). Then `auto`: on iff stderr is a TTY.
	pub fn from_env() -> Self {
		if std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty()) {
			return ColorMode::Off;
		}
		match std::env::var("CCP_LOG_COLOR")
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
		Self { writer, mode, first: true, result: Ok(()) }
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
	fn write_field(&mut self, name: &str, value: &dyn fmt::Debug) -> fmt::Result {
		self.write_separator()?;

		// The "message" field of an event is rendered without a key prefix
		// by the default formatter. Keep that contract.
		if name == "message" {
			return write!(self.writer, "{:?}", value);
		}

		write!(self.writer, "{}=", name)?;

		// Non-color path: short-circuit to default Debug rendering.
		if matches!(self.mode, ColorMode::Off) {
			return write!(self.writer, "{:?}", value);
		}

		// We need the value as a string to (a) decide how to color it and
		// (b) apply the style. Format into a small buffer, strip surrounding
		// quotes that Debug adds for &str, then color.
		let mut buf = String::new();
		write!(&mut buf, "{:?}", value)?;
		let style = style_for(name, &buf);
		match style {
			Some(s) => write!(self.writer, "{}", s.paint(buf.as_str())),
			None => write!(self.writer, "{}", buf),
		}
	}
}

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
		"request_id" | "session_id" | "pid" => Some(palette_style(debug_text)),
		"finish_reason" => match debug_text.trim_matches('"') {
			"tool_calls" => Some(Style::new().fg(Color::Fixed(178)).bold()),
			"stop" => Some(Style::new().fg(Color::Fixed(70))),
			_ => None,
		},
		_ => None,
	}
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
	fn unknown_fields_uncolored() {
		assert!(style_for("model", "\"opus\"").is_none());
		assert!(style_for("duration_ms", "1234").is_none());
	}

	#[test]
	fn no_color_env_disables() {
		// Save and restore env for hermetic test.
		let prev_no = std::env::var("NO_COLOR").ok();
		let prev_ccp = std::env::var("CCP_LOG_COLOR").ok();
		// SAFETY: tests run single-threaded by default per crate test harness;
		// these env reads happen before the formatter is initialized in main.
		unsafe {
			std::env::set_var("NO_COLOR", "1");
			std::env::set_var("CCP_LOG_COLOR", "always");
		}
		assert_eq!(ColorMode::from_env(), ColorMode::Off);
		unsafe {
			match prev_no {
				Some(v) => std::env::set_var("NO_COLOR", v),
				None => std::env::remove_var("NO_COLOR"),
			}
			match prev_ccp {
				Some(v) => std::env::set_var("CCP_LOG_COLOR", v),
				None => std::env::remove_var("CCP_LOG_COLOR"),
			}
		}
	}
}
