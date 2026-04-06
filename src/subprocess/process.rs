use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::config::Config;
use crate::subprocess::ndjson::{self, ResultMessage};

/// Removes a directory when dropped.
struct DirGuard(Option<PathBuf>);

impl Drop for DirGuard {
	fn drop(&mut self) {
		if let Some(ref dir) = self.0 {
			if let Err(e) = std::fs::remove_dir_all(dir) {
				warn!(path = ?dir, error = %e, "Failed to clean up request config dir");
			}
		}
	}
}

/// Events emitted by the subprocess to the route handler.
#[derive(Debug)]
pub enum SubprocessEvent {
	/// Model name extracted from assistant or message_start.
	Model(String),
	/// A text_delta content fragment.
	ContentDelta(String),
	/// The final result message.
	Result(Box<ResultMessage>),
	/// An error occurred.
	Error(String),
}

/// Spawn and manage a single claude subprocess.
/// Reads NDJSON from stdout, sends SubprocessEvents via tx.
pub async fn run_subprocess(
	config: &Config,
	request_id: &str,
	cli_args: Vec<String>,
	tx: mpsc::Sender<SubprocessEvent>,
) {
	let start = Instant::now();
	let mut ttft_logged = false;

	debug!(args = ?cli_args, "Spawning subprocess");

	// Each request gets its own config dir so concurrent subprocesses don't
	// interfere, and stale OAuth caches don't survive across requests.
	let request_config_dir = config.create_request_config_dir(request_id);
	let _dir_guard = DirGuard(request_config_dir.clone());

	let mut cmd = tokio::process::Command::new(&config.claude_path);
	cmd.args(&cli_args)
		.stdin(Stdio::null())
		.stdout(Stdio::piped())
		.stderr(Stdio::piped())
		.kill_on_drop(true)
		.env_remove("ANTHROPIC_API_KEY")
		.env_remove("ANTHROPIC_BASE_URL")
		.env_remove("ANTHROPIC_AUTH_TOKEN")
		.current_dir(config.resolved_working_dir());

	if let Some(ref dir) = request_config_dir {
		cmd.env("CLAUDE_CONFIG_DIR", dir.to_str().unwrap_or(""));
	}

	let mut child = match cmd.spawn() {
		Ok(child) => child,
		Err(e) => {
			let msg = if e.kind() == std::io::ErrorKind::NotFound {
				format!(
					"claude CLI not found at '{}'. Install with: npm install -g @anthropic-ai/claude-code",
					config.claude_path
				)
			} else {
				format!("Failed to spawn claude: {}", e)
			};
			error!("{}", msg);
			let _ = tx.send(SubprocessEvent::Error(msg)).await;
			return;
		}
	};

	let pid = child.id().unwrap_or(0);
	info!(pid, "Subprocess started");

	let stdout = child.stdout.take().expect("stdout not captured");
	let stderr = child.stderr.take().expect("stderr not captured");

	let mut stdout_lines = BufReader::new(stdout).lines();
	let mut stderr_lines = BufReader::new(stderr).lines();
	let mut stderr_buf: VecDeque<String> = VecDeque::new();
	let mut got_result = false;
	let mut line_count: u64 = 0;
	let mut chunk_count: u64 = 0;

	let timeout_dur = Duration::from_secs(config.timeout);
	let inactivity = tokio::time::sleep(timeout_dur);
	tokio::pin!(inactivity);
	let progress = tokio::time::sleep(Duration::from_secs(30));
	tokio::pin!(progress);

	loop {
		tokio::select! {
			line = stdout_lines.next_line() => {
				match line {
					Ok(Some(line)) => {
						inactivity.as_mut().reset(tokio::time::Instant::now() + timeout_dur);
						line_count += 1;

						if let Some(msg) = ndjson::parse_line(&line) {
							for event in ndjson::process_message(msg) {
								if matches!(&event, SubprocessEvent::ContentDelta(_)) {
									if !ttft_logged {
										let ttft = start.elapsed().as_secs_f64() * 1000.0;
										info!(pid, ttft_ms = format!("{:.0}", ttft), "First token");
										ttft_logged = true;
									}
									chunk_count += 1;
								}
								if matches!(&event, SubprocessEvent::Result(_)) {
									got_result = true;
								}
								if tx.send(event).await.is_err() {
									warn!(pid, "Client disconnected, killing subprocess");
									let _ = child.kill().await;
									return;
								}
							}
						}
					}
					Ok(None) => break, // stdout closed
					Err(e) => {
						error!(pid, error = %e, "Error reading stdout");
						break;
					}
				}
			}
			line = stderr_lines.next_line() => {
				match line {
					Ok(Some(line)) => {
						inactivity.as_mut().reset(tokio::time::Instant::now() + timeout_dur);
						debug!(pid, line = %line, "stderr");
						if stderr_buf.len() >= 50 {
							stderr_buf.pop_front();
						}
						stderr_buf.push_back(line);
					}
					Ok(None) => {} // stderr closed, wait for stdout
					Err(_) => {}
				}
			}
			() = &mut inactivity => {
				warn!(pid, elapsed_s = start.elapsed().as_secs(), "Inactivity timeout");
				let _ = tx.send(SubprocessEvent::Error("Inactivity timeout".into())).await;
				let _ = child.kill().await;
				return;
			}
			() = &mut progress => {
				info!(
					pid,
					elapsed_s = start.elapsed().as_secs(),
					lines = line_count,
					chunks = chunk_count,
					"Still running"
				);
				progress.as_mut().reset(tokio::time::Instant::now() + Duration::from_secs(30));
			}
		}
	}

	// Wait for process exit.
	let exit_code = match child.wait().await {
		Ok(status) => status.code().unwrap_or(-1),
		Err(e) => {
			error!(pid, error = %e, "Error waiting for subprocess");
			-1
		}
	};

	let elapsed = start.elapsed().as_secs_f64();
	info!(pid, exit_code, elapsed_s = format!("{:.2}", elapsed), "Subprocess exited");

	if !got_result {
		let stderr_text = if stderr_buf.is_empty() {
			format!(
				"Process exited with code {} (no result; {} lines, {} chunks)",
				exit_code, line_count, chunk_count
			)
		} else {
			format!(
				"Process exited with code {}: {}",
				exit_code,
				stderr_buf.make_contiguous().join("\n")
			)
		};
		let _ = tx.send(SubprocessEvent::Error(stderr_text)).await;
	}
}
