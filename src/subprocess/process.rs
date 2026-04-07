use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
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
/// The prompt is piped via stdin; NDJSON is read from stdout.
pub async fn run_subprocess(
	config: &Config,
	request_id: &str,
	cli_args: Vec<String>,
	prompt: String,
	tx: mpsc::Sender<SubprocessEvent>,
) {
	let start = Instant::now();
	let mut ttft_logged = false;

	debug!(args = ?cli_args, prompt_bytes = prompt.len(), "Spawning subprocess");

	// Each request gets its own config dir so concurrent subprocesses don't
	// interfere, and stale OAuth caches don't survive across requests.
	let request_config_dir = config.create_request_config_dir(request_id);
	let _dir_guard = DirGuard(request_config_dir.clone());

	let mut cmd = tokio::process::Command::new(&config.claude_path);
	cmd.args(&cli_args)
		.stdin(Stdio::piped())
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

	// Write the prompt to stdin concurrently with reading stdout/stderr.
	// This avoids deadlock when the prompt exceeds the OS pipe buffer (~64KB):
	// the child may write to stdout/stderr before consuming all of stdin.
	let stdin = child.stdin.take().expect("stdin not captured");
	let stdin_tx = tx.clone();
	let stdin_task = tokio::spawn(async move {
		let mut stdin = stdin;
		if let Err(e) = stdin.write_all(prompt.as_bytes()).await {
			error!(error = %e, "Failed to write prompt to stdin");
			let _ = stdin_tx
				.send(SubprocessEvent::Error(format!(
					"Failed to write prompt to subprocess stdin: {}",
					e
				)))
				.await;
		}
		// stdin is dropped here, closing the pipe and signaling EOF.
	});

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
								// Enrich error results that have no message with stderr.
								let event = if let SubprocessEvent::Result(mut result) = event {
									got_result = true;
									if result.is_error.unwrap_or(false) && result.result.is_none() && !stderr_buf.is_empty() {
										result.result = Some(stderr_buf.make_contiguous().join("\n"));
									}
									SubprocessEvent::Result(result)
								} else {
									event
								};
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

	// Ensure the stdin writer has finished (it may already be done).
	stdin_task.abort();

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
