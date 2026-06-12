use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::mpsc::{SyncSender, sync_channel};

pub const DEFAULT_LOG_MAX_BYTES: u64 = 64 * 1024 * 1024;
pub const DEFAULT_LOG_BACKUPS: usize = 5;

/// Bound on the pending-log queue. Generous, but caps memory if the disk writer
/// falls behind under heavy load; excess records are dropped with a warning
/// rather than blocking a request or growing without limit.
const LOG_QUEUE_CAPACITY: usize = 8192;

/// How long `Drop` waits for the writer to finish flushing before giving up, so
/// a wedged disk cannot block process/teardown indefinitely.
const SHUTDOWN_FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// A single log record handed to the writer thread. Holds exactly one copy of
/// the formatted `record` (header + content + separator); the writer reuses it
/// for every target, so a large payload is not cloned multiple times per entry.
struct LogMsg {
    session_id: String,
    request_id: String,
    record: String,
}

/// Logs conversation prompts and responses to a file or stderr.
///
/// Writes happen on a dedicated OS thread fed by a bounded channel, so logging
/// never blocks a Tokio worker on disk I/O and concurrent requests do not
/// serialize on a shared mutex during the write. Dropping the log closes the
/// queue and waits (bounded) for the writer to flush pending records.
pub struct ConversationLog {
    tx: Option<SyncSender<LogMsg>>,
    /// Signalled by the writer thread when its loop exits (channel drained).
    /// Wrapped in a Mutex so `ConversationLog` stays `Sync` (a bare `Receiver`
    /// is `Send` but not `Sync`, and this type is shared via `Arc` in AppState).
    done: Mutex<std::sync::mpsc::Receiver<()>>,
}

impl Drop for ConversationLog {
    fn drop(&mut self) {
        // Close the channel so the writer loop ends after draining, then wait a
        // bounded time for it to finish. A wedged disk must not hang shutdown.
        self.tx.take();
        let done = self.done.get_mut().unwrap_or_else(|e| e.into_inner());
        if done.recv_timeout(SHUTDOWN_FLUSH_TIMEOUT).is_err() {
            tracing::warn!("conversation log writer did not flush within timeout on shutdown");
        }
    }
}

enum Target {
    File(RotatingFile),
    Directory(LogDirectory),
    Stderr,
}

impl Target {
    fn write(&mut self, msg: &LogMsg) {
        match self {
            Target::File(f) => {
                if let Err(e) = f.write_record(&msg.record) {
                    tracing::warn!("failed to write conversation log: {e}");
                }
            }
            Target::Directory(dir) => {
                if let Err(e) = dir.write_record(&msg.session_id, &msg.request_id, &msg.record) {
                    tracing::warn!("failed to write conversation log: {e}");
                }
            }
            Target::Stderr => {
                // `record` is already "header\ncontent\nseparator\n\n".
                tracing::info!("{}", msg.record.trim_end());
            }
        }
    }
}

/// Spawn the writer thread that owns `target` and drains the queue until all
/// senders are dropped. Returns the sender plus a receiver that is signalled
/// when the writer's loop exits (used for a bounded flush on drop). A panic in a
/// single write is caught so it cannot silently kill the writer.
fn spawn_writer(mut target: Target) -> (SyncSender<LogMsg>, std::sync::mpsc::Receiver<()>) {
    let (tx, rx) = sync_channel::<LogMsg>(LOG_QUEUE_CAPACITY);
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    let spawned = std::thread::Builder::new()
        .name("omni-conversation-log".into())
        .spawn(move || {
            for msg in rx {
                // One wedged/oversized record must not kill the writer thread.
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| target.write(&msg)))
                    .is_err()
                {
                    tracing::warn!("conversation log writer recovered from a write panic");
                }
            }
            let _ = done_tx.send(());
        });
    if spawned.is_err() {
        // Could not spawn the writer: signal "done" immediately so Drop never
        // blocks, and let sends fail/drop. Logging silently degrades but the
        // request path is unaffected.
        tracing::warn!("failed to spawn conversation log writer thread; logging disabled");
        let (immediate_tx, immediate_rx) = std::sync::mpsc::channel::<()>();
        let _ = immediate_tx.send(());
        return (tx, immediate_rx);
    }
    (tx, done_rx)
}

/// Directory logging: one append-only file per session id under `path`.
///
/// Note: unlike [`RotatingFile`], per-session files are intentionally NOT
/// size-capped or rotated. For normal use a session file is bounded by the
/// conversation's length, so this is acceptable; a very long-running session
/// (e.g. an agent loop reusing one session id) can grow its file without bound.
/// If that becomes a problem, prefer single-file mode (`--log-file`), which
/// honours `--log-max-bytes`/backups.
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
        let (tx, done) = spawn_writer(Target::File(file));
        Ok(Self {
            tx: Some(tx),
            done: Mutex::new(done),
        })
    }

    pub fn to_dir(path: &Path) -> Result<Self, std::io::Error> {
        std::fs::create_dir_all(path)?;
        let (tx, done) = spawn_writer(Target::Directory(LogDirectory {
            path: path.to_path_buf(),
        }));
        Ok(Self {
            tx: Some(tx),
            done: Mutex::new(done),
        })
    }

    pub fn to_stderr() -> Self {
        let (tx, done) = spawn_writer(Target::Stderr);
        Self {
            tx: Some(tx),
            done: Mutex::new(done),
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

        let msg = LogMsg {
            session_id: session_id.to_string(),
            request_id: request_id.to_string(),
            record,
        };

        // Never block a request on logging: if the writer is behind (queue full)
        // or gone, drop the record with a warning.
        if let Some(tx) = self.tx.as_ref()
            && let Err(e) = tx.try_send(msg)
        {
            tracing::warn!("conversation log queue full or closed, dropping record: {e}");
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

        // Surface a clean error instead of panicking if the file handle is somehow
        // absent (e.g. a prior rotate() failed to reopen). The writer thread's
        // catch_unwind would contain a panic, but returning Err is tidier and lets
        // the caller log it.
        let Some(file) = self.file.as_mut() else {
            return Err(std::io::Error::other("rotating log file is not open"));
        };
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
        let dir = std::env::temp_dir().join(format!("omni-log-test-{}", uuid::Uuid::new_v4()));
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

        // Drop the log to flush the writer thread before reading files.
        drop(log);

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
        let dir = std::env::temp_dir().join(format!("omni-log-test-{}", uuid::Uuid::new_v4()));
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

        drop(log);
        assert!(std::fs::metadata(&path).unwrap().len() > 220);
        assert!(!path.with_file_name("conversations.log.1").exists());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn directory_logging_writes_one_file_per_sanitized_session() {
        let dir = std::env::temp_dir().join(format!("omni-log-test-{}", uuid::Uuid::new_v4()));
        let log = ConversationLog::to_dir(&dir).unwrap();

        log.log("x:alpha beta", "reqone", ">>>", "First", "one");
        log.log("x:alpha beta", "reqtwo", "<<<", "Second", "two");
        log.log("u:other/session", "reqthree", ">>>", "Other", "three");

        drop(log);
        let alpha = std::fs::read_to_string(dir.join("x_alpha_beta.log")).unwrap();
        assert!(alpha.contains("request=reqone"));
        assert!(alpha.contains("request=reqtwo"));
        assert!(alpha.contains("session=x:alpha beta"));
        assert!(dir.join("u_other_session.log").exists());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn directory_logging_uses_request_id_for_unknown_session() {
        let dir = std::env::temp_dir().join(format!("omni-log-test-{}", uuid::Uuid::new_v4()));
        let log = ConversationLog::to_dir(&dir).unwrap();

        log.log("-", "reqsolo", ">>>", "Only", "payload");

        drop(log);
        assert!(dir.join("request-reqsolo.log").exists());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn session_filename_falls_back_for_empty_or_unsafe_ids() {
        assert_eq!(session_filename("-"), "-");
        assert_eq!(session_filename(":::"), "unknown-session");
    }
}
