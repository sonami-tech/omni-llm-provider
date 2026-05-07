use std::collections::VecDeque;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{ChildStderr, ChildStdout};
use tokio::sync::mpsc;
use tracing::{Instrument, debug, error, info, warn};

use crate::config::Config;
use crate::subprocess::ndjson::{self, ResultMessage};

const STDERR_BUF_CAP: usize = 50;
const POST_EXIT_DRAIN: Duration = Duration::from_millis(200);
const PROGRESS_INTERVAL: Duration = Duration::from_secs(30);

type StdoutLines = Lines<BufReader<ChildStdout>>;
type StderrLines = Lines<BufReader<ChildStderr>>;

/// Yield the next line from `lines` if open, or park forever if `None`.
/// Used so a closed stream's `select!` arm becomes inert without spinning.
async fn next_line_or_pending<R>(lines: &mut Option<Lines<R>>) -> std::io::Result<Option<String>>
where
	R: tokio::io::AsyncBufRead + Unpin,
{
	match lines.as_mut() {
		Some(l) => l.next_line().await,
		None => std::future::pending().await,
	}
}

fn push_stderr(buf: &mut VecDeque<String>, line: String) {
	if buf.len() >= STDERR_BUF_CAP {
		buf.pop_front();
	}
	buf.push_back(line);
}

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
	let stdin_task = tokio::spawn(
		async move {
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
		}
		.instrument(tracing::Span::current()),
	);

	let stdout = child.stdout.take().expect("stdout not captured");
	let stderr = child.stderr.take().expect("stderr not captured");

	let mut stdout_lines: Option<StdoutLines> = Some(BufReader::new(stdout).lines());
	let mut stderr_lines: Option<StderrLines> = Some(BufReader::new(stderr).lines());
	let mut stderr_buf: VecDeque<String> = VecDeque::new();
	let mut got_result = false;
	let mut line_count: u64 = 0;
	let mut chunk_count: u64 = 0;

	let timeout_dur = Duration::from_secs(config.timeout);
	let inactivity = tokio::time::sleep(timeout_dur);
	tokio::pin!(inactivity);
	let progress = tokio::time::sleep(PROGRESS_INTERVAL);
	tokio::pin!(progress);

	let exit_code: i32 = 'outer: loop {
		tokio::select! {
			status = child.wait() => {
				let code = match status {
					Ok(s) => s.code().unwrap_or(-1),
					Err(e) => {
						error!(pid, error = %e, "Error waiting for subprocess");
						-1
					}
				};
				// Drain any buffered stdout/stderr that arrived just before
				// exit — bounded so descendants holding the pipe can't stall
				// the drain.
				let drain = tokio::time::sleep(POST_EXIT_DRAIN);
				tokio::pin!(drain);
				loop {
					tokio::select! {
						() = &mut drain => break,
						line = next_line_or_pending(&mut stdout_lines) => match line {
							Ok(Some(line)) => {
								line_count += 1;
								if !dispatch_stdout_line(
									&line, &tx, &mut stderr_buf, &mut got_result, pid,
								).await {
									break 'outer code;
								}
							}
							_ => { stdout_lines = None; }
						},
						line = next_line_or_pending(&mut stderr_lines) => match line {
							Ok(Some(line)) => push_stderr(&mut stderr_buf, line),
							_ => { stderr_lines = None; }
						},
					}
					if stdout_lines.is_none() && stderr_lines.is_none() { break; }
				}
				break 'outer code;
			}
			line = next_line_or_pending(&mut stdout_lines) => {
				match line {
					Ok(Some(line)) => {
						inactivity.as_mut().reset(tokio::time::Instant::now() + timeout_dur);
						line_count += 1;
						if !dispatch_stdout_line_with_ttft(
							&line, &tx, &mut stderr_buf, &mut got_result,
							&mut ttft_logged, &mut chunk_count, pid, start,
						).await {
							warn!(pid, "Client disconnected, killing subprocess");
							teardown(stdin_task, &mut child).await;
							return;
						}
					}
					Ok(None) => { stdout_lines = None; }
					Err(e) => {
						error!(pid, error = %e, "Error reading stdout");
						stdout_lines = None;
					}
				}
			}
			line = next_line_or_pending(&mut stderr_lines) => {
				match line {
					Ok(Some(line)) => {
						inactivity.as_mut().reset(tokio::time::Instant::now() + timeout_dur);
						debug!(pid, line = %line, "stderr");
						push_stderr(&mut stderr_buf, line);
					}
					Ok(None) | Err(_) => { stderr_lines = None; }
				}
			}
			() = &mut inactivity => {
				warn!(pid, elapsed_s = start.elapsed().as_secs(), "Inactivity timeout");
				let _ = tx.send(SubprocessEvent::Error("Inactivity timeout".into())).await;
				teardown(stdin_task, &mut child).await;
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
				progress.as_mut().reset(tokio::time::Instant::now() + PROGRESS_INTERVAL);
			}
		}
	};

	stdin_task.abort();
	let _ = stdin_task.await;

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

/// Parse one stdout NDJSON line, enrich any error result with stderr, and
/// forward each derived event. Returns `false` if the receiver has dropped.
async fn dispatch_stdout_line(
	line: &str,
	tx: &mpsc::Sender<SubprocessEvent>,
	stderr_buf: &mut VecDeque<String>,
	got_result: &mut bool,
	pid: u32,
) -> bool {
	let Some(msg) = ndjson::parse_line(line) else {
		return true;
	};
	for event in ndjson::process_message(msg) {
		let event = if let SubprocessEvent::Result(mut result) = event {
			*got_result = true;
			result.enrich_with_stderr(stderr_buf, pid, line);
			SubprocessEvent::Result(result)
		} else {
			event
		};
		if tx.send(event).await.is_err() {
			return false;
		}
	}
	true
}

/// Same as [`dispatch_stdout_line`] but also tracks TTFT logging and content
/// chunk counts for the main I/O loop. Returns `false` on receiver drop.
#[allow(clippy::too_many_arguments)]
async fn dispatch_stdout_line_with_ttft(
	line: &str,
	tx: &mpsc::Sender<SubprocessEvent>,
	stderr_buf: &mut VecDeque<String>,
	got_result: &mut bool,
	ttft_logged: &mut bool,
	chunk_count: &mut u64,
	pid: u32,
	start: Instant,
) -> bool {
	let Some(msg) = ndjson::parse_line(line) else {
		return true;
	};
	for event in ndjson::process_message(msg) {
		if matches!(&event, SubprocessEvent::ContentDelta(_)) {
			if !*ttft_logged {
				let ttft = start.elapsed().as_secs_f64() * 1000.0;
				info!(pid, ttft_ms = format!("{:.0}", ttft), "First token");
				*ttft_logged = true;
			}
			*chunk_count += 1;
		}
		let event = if let SubprocessEvent::Result(mut result) = event {
			*got_result = true;
			result.enrich_with_stderr(stderr_buf, pid, line);
			SubprocessEvent::Result(result)
		} else {
			event
		};
		if tx.send(event).await.is_err() {
			return false;
		}
	}
	true
}

/// Drop stdin writer and reap the child. Used on every early-return path so
/// the kernel doesn't accumulate zombie subprocesses.
async fn teardown(stdin_task: tokio::task::JoinHandle<()>, child: &mut tokio::process::Child) {
	stdin_task.abort();
	let _ = stdin_task.await;
	let _ = child.kill().await;
	let _ = child.wait().await;
}
