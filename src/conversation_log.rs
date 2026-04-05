use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

/// Logs conversation prompts and responses to a file or stderr.
pub struct ConversationLog {
	target: Mutex<Target>,
}

enum Target {
	File(std::fs::File),
	Stderr,
}

impl ConversationLog {
	pub fn to_file(path: &Path) -> Result<Self, std::io::Error> {
		let file = std::fs::OpenOptions::new()
			.create(true)
			.append(true)
			.open(path)?;
		Ok(Self {
			target: Mutex::new(Target::File(file)),
		})
	}

	pub fn to_stderr() -> Self {
		Self {
			target: Mutex::new(Target::Stderr),
		}
	}

	pub fn log(&self, request_id: &str, direction: &str, label: &str, content: &str) {
		let timestamp = timestamp_now();
		let header = format!("[{}] {} {} {}", timestamp, request_id, direction, label);
		let separator = "-".repeat(header.len().min(72));

		let mut target = self.target.lock().unwrap_or_else(|e| e.into_inner());
		match *target {
			Target::File(ref mut f) => {
				let _ = writeln!(f, "{}\n{}\n{}\n", header, content, separator);
				let _ = f.flush();
			}
			Target::Stderr => {
				tracing::info!("{}\n{}\n{}", header, content, separator);
			}
		}
	}
}

fn timestamp_now() -> String {
	use std::time::{SystemTime, UNIX_EPOCH};
	let secs = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.unwrap_or_default()
		.as_secs();
	let hours = (secs % 86400) / 3600;
	let minutes = (secs % 3600) / 60;
	let seconds = secs % 60;
	format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}
