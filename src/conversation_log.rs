use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const DEFAULT_LOG_MAX_BYTES: u64 = 64 * 1024 * 1024;
pub const DEFAULT_LOG_BACKUPS: usize = 5;

/// Logs conversation prompts and responses to a file or stderr.
pub struct ConversationLog {
	target: Mutex<Target>,
}

enum Target {
	File(RotatingFile),
	Directory(LogDirectory),
	Stderr,
}

struct LogDirectory {
	path: PathBuf,
}

struct RotatingFile {
	path: PathBuf,
	file: Option<std::fs::File>,
	current_len: u64,
	max_bytes: u64,
	backups: usize,
}

impl ConversationLog {
	pub fn to_file(path: &Path, max_bytes: u64, backups: usize) -> Result<Self, std::io::Error> {
		let file = RotatingFile::open(path, max_bytes, backups)?;
		Ok(Self {
			target: Mutex::new(Target::File(file)),
		})
	}

	pub fn to_dir(path: &Path) -> Result<Self, std::io::Error> {
		std::fs::create_dir_all(path)?;
		Ok(Self {
			target: Mutex::new(Target::Directory(LogDirectory {
				path: path.to_path_buf(),
			})),
		})
	}

	pub fn to_stderr() -> Self {
		Self {
			target: Mutex::new(Target::Stderr),
		}
	}

	pub fn log(
		&self,
		session_id: &str,
		request_id: &str,
		direction: &str,
		label: &str,
		content: &str,
	) {
		let timestamp = crate::time_util::time_of_day_now();
		let header = format!(
			"[{}] session={} request={} {} {}",
			timestamp, session_id, request_id, direction, label
		);
		let separator = "-".repeat(header.len().min(72));
		let record = format!("{}\n{}\n{}\n\n", header, content, separator);

		let mut target = self.target.lock().unwrap_or_else(|e| e.into_inner());
		match *target {
			Target::File(ref mut f) => {
				if let Err(e) = f.write_record(&record) {
					tracing::warn!("failed to write conversation log: {e}");
				}
			}
			Target::Directory(ref dir) => {
				if let Err(e) = dir.write_record(session_id, request_id, &record) {
					tracing::warn!("failed to write conversation log: {e}");
				}
			}
			Target::Stderr => {
				tracing::info!("{}\n{}\n{}", header, content, separator);
			}
		}
	}
}

impl LogDirectory {
	fn write_record(
		&self,
		session_id: &str,
		request_id: &str,
		record: &str,
	) -> Result<(), std::io::Error> {
		let file_stem = if session_id == "-" {
			format!("request-{}", request_id)
		} else {
			session_filename(session_id)
		};
		let path = self.path.join(format!("{file_stem}.log"));
		let mut file = open_append(&path)?;
		file.write_all(record.as_bytes())?;
		file.flush()
	}
}

impl RotatingFile {
	fn open(path: &Path, max_bytes: u64, backups: usize) -> Result<Self, std::io::Error> {
		if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
			std::fs::create_dir_all(parent)?;
		}

		let file = open_append(path)?;
		let current_len = file.metadata().map(|m| m.len()).unwrap_or(0);
		Ok(Self {
			path: path.to_path_buf(),
			file: Some(file),
			current_len,
			max_bytes,
			backups,
		})
	}

	fn write_record(&mut self, record: &str) -> Result<(), std::io::Error> {
		let record_len = record.len() as u64;
		if self.should_rotate(record_len) {
			self.rotate()?;
		}

		let file = self
			.file
			.as_mut()
			.expect("rotating log file should be open before write");
		file.write_all(record.as_bytes())?;
		file.flush()?;
		self.current_len = self.current_len.saturating_add(record_len);
		Ok(())
	}

	fn should_rotate(&self, next_len: u64) -> bool {
		self.max_bytes > 0
			&& self.current_len > 0
			&& self.current_len.saturating_add(next_len) > self.max_bytes
	}

	fn rotate(&mut self) -> Result<(), std::io::Error> {
		if let Some(mut file) = self.file.take() {
			file.flush()?;
		}

		if self.backups == 0 {
			let _ = std::fs::remove_file(&self.path);
		} else {
			let oldest = backup_path(&self.path, self.backups);
			let _ = std::fs::remove_file(oldest);

			for index in (1..self.backups).rev() {
				let src = backup_path(&self.path, index);
				if src.exists() {
					let dst = backup_path(&self.path, index + 1);
					let _ = std::fs::remove_file(&dst);
					std::fs::rename(src, dst)?;
				}
			}

			if self.path.exists() {
				let first = backup_path(&self.path, 1);
				let _ = std::fs::remove_file(&first);
				std::fs::rename(&self.path, first)?;
			}
		}

		self.file = Some(open_append(&self.path)?);
		self.current_len = 0;
		Ok(())
	}
}

fn open_append(path: &Path) -> Result<std::fs::File, std::io::Error> {
	std::fs::OpenOptions::new()
		.create(true)
		.append(true)
		.open(path)
}

fn backup_path(path: &Path, index: usize) -> PathBuf {
	let mut name = path
		.file_name()
		.unwrap_or_else(|| OsStr::new("conversations.log"))
		.to_os_string();
	name.push(format!(".{index}"));
	path.with_file_name(name)
}

fn session_filename(session_id: &str) -> String {
	let cleaned: String = session_id
		.chars()
		.map(|c| {
			if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
				c
			} else {
				'_'
			}
		})
		.collect();
	let trimmed = cleaned.trim_matches('_');
	if trimmed.is_empty() {
		"unknown-session".to_string()
	} else {
		trimmed.chars().take(160).collect()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn rotates_file_when_max_bytes_is_exceeded() {
		let dir = std::env::temp_dir().join(format!("ccp-log-test-{}", uuid::Uuid::new_v4()));
		let path = dir.join("conversations.log");
		let log = ConversationLog::to_file(&path, 220, 2).unwrap();

		for i in 0..8 {
			log.log(
				"h:session",
				"reqtest",
				">>>",
				"Test entry",
				&format!("payload-{i}-{}", "x".repeat(80)),
			);
		}

		let active_len = std::fs::metadata(&path).unwrap().len();
		assert!(active_len > 0);
		assert!(active_len <= 220);
		assert!(path.with_file_name("conversations.log.1").exists());
		assert!(path.with_file_name("conversations.log.2").exists());
		assert!(!path.with_file_name("conversations.log.3").exists());

		let _ = std::fs::remove_dir_all(dir);
	}

	#[test]
	fn max_bytes_zero_disables_rotation() {
		let dir = std::env::temp_dir().join(format!("ccp-log-test-{}", uuid::Uuid::new_v4()));
		let path = dir.join("conversations.log");
		let log = ConversationLog::to_file(&path, 0, 2).unwrap();

		for i in 0..4 {
			log.log(
				"h:session",
				"reqtest",
				">>>",
				"Test entry",
				&format!("payload-{i}-{}", "x".repeat(80)),
			);
		}

		assert!(std::fs::metadata(&path).unwrap().len() > 220);
		assert!(!path.with_file_name("conversations.log.1").exists());

		let _ = std::fs::remove_dir_all(dir);
	}

	#[test]
	fn directory_logging_writes_one_file_per_sanitized_session() {
		let dir = std::env::temp_dir().join(format!("ccp-log-test-{}", uuid::Uuid::new_v4()));
		let log = ConversationLog::to_dir(&dir).unwrap();

		log.log("x:alpha beta", "reqone", ">>>", "First", "one");
		log.log("x:alpha beta", "reqtwo", "<<<", "Second", "two");
		log.log("u:other/session", "reqthree", ">>>", "Other", "three");

		let alpha = std::fs::read_to_string(dir.join("x_alpha_beta.log")).unwrap();
		assert!(alpha.contains("request=reqone"));
		assert!(alpha.contains("request=reqtwo"));
		assert!(alpha.contains("session=x:alpha beta"));
		assert!(dir.join("u_other_session.log").exists());

		let _ = std::fs::remove_dir_all(dir);
	}

	#[test]
	fn directory_logging_uses_request_id_for_unknown_session() {
		let dir = std::env::temp_dir().join(format!("ccp-log-test-{}", uuid::Uuid::new_v4()));
		let log = ConversationLog::to_dir(&dir).unwrap();

		log.log("-", "reqsolo", ">>>", "Only", "payload");

		assert!(dir.join("request-reqsolo.log").exists());

		let _ = std::fs::remove_dir_all(dir);
	}

	#[test]
	fn session_filename_falls_back_for_empty_or_unsafe_ids() {
		assert_eq!(session_filename("-"), "-");
		assert_eq!(session_filename(":::"), "unknown-session");
	}
}
