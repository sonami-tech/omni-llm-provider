use std::path::PathBuf;

use clap::Parser;

const CREDENTIALS_FILENAME: &str = ".credentials.json";

#[derive(Parser, Clone, Debug)]
#[command(
	name = "claude-code-provider",
	version,
	about = "OpenAI-compatible API proxy for Claude Max accounts"
)]
pub struct Config {
	/// Listen port.
	#[arg(short = 'p', long, default_value = "18321", env = "CCP_PORT")]
	pub port: u16,

	/// Listen address.
	#[arg(short = 'H', long, default_value = "127.0.0.1", env = "CCP_HOST")]
	pub host: String,

	/// Max in-flight requests (semaphore size).
	#[arg(short = 'c', long, default_value = "5", env = "CCP_MAX_CONCURRENT")]
	pub max_concurrent: usize,

	/// Per-request timeout in seconds. (v1 legacy: only enforced on the
	/// removed subprocess path. v2 uses a fixed 600s upstream timeout.)
	#[arg(short = 't', long, default_value = "600", env = "CCP_TIMEOUT")]
	pub timeout: u64,

	/// Max time a request waits in the concurrency queue (seconds).
	#[arg(short = 'q', long, default_value = "60", env = "CCP_QUEUE_TIMEOUT")]
	pub queue_timeout: u64,

	/// Path to the `claude` CLI binary, used at startup only to verify
	/// installation and login state. v2 does not invoke it per request.
	#[arg(long, default_value = "claude", env = "CCP_CLAUDE_PATH")]
	pub claude_path: String,

	/// Data directory for the stats DB (and v1 config isolation).
	#[arg(long, env = "CCP_DATA_DIR")]
	pub data_dir: Option<PathBuf>,

	/// (v1 legacy) Working directory for subprocesses. Unused in v2.
	#[arg(long, env = "CCP_WORKING_DIR")]
	pub working_dir: Option<PathBuf>,

	/// (v1 legacy) Disable per-request config isolation. Unused in v2.
	#[arg(long, env = "CCP_NO_ISOLATE")]
	pub no_isolate: bool,

	/// API keys (comma-separated). If set, requests must include a valid key.
	#[arg(long, env = "CCP_API_KEYS", value_delimiter = ',')]
	pub api_keys: Vec<String>,

	/// File containing API keys (one per line, # comments allowed).
	#[arg(long, env = "CCP_API_KEYS_FILE")]
	pub api_keys_file: Option<PathBuf>,

	/// Disable authentication entirely.
	#[arg(long, env = "CCP_NO_AUTH")]
	pub no_auth: bool,

	/// Disable tool/function call passthrough.
	#[arg(long, env = "CCP_NO_TOOL_PASSTHROUGH")]
	pub no_tool_passthrough: bool,

	/// TOML file with text replacement rules.
	#[arg(long, env = "CCP_REPLACE_RULES")]
	pub replace_rules: Option<PathBuf>,

	/// Log full prompts and responses.
	#[arg(long, env = "CCP_LOG_CONVERSATIONS")]
	pub log_conversations: bool,

	/// File to write conversation logs to (implies --log-conversations).
	#[arg(long, env = "CCP_LOG_FILE")]
	pub log_file: Option<PathBuf>,

	/// Skip prepending the canonical Claude Code system identifier block
	/// to outbound requests. The preamble is required for opus/sonnet
	/// calls to bypass Anthropic's OAuth subscription gate; disable only
	/// if the consumer is providing its own equivalent system prompt or
	/// for upstream debugging.
	#[arg(long, env = "CCP_NO_PREAMBLE")]
	pub no_preamble: bool,

	/// Enable debug logging.
	#[arg(short = 'v', long)]
	pub verbose: bool,
}

impl Config {
	pub fn resolved_data_dir(&self) -> PathBuf {
		self.data_dir.clone().unwrap_or_else(|| {
			dirs::data_dir()
				.expect("Could not determine data directory")
				.join("claude-code-provider")
		})
	}

	pub fn isolated_config_dir(&self) -> PathBuf {
		self.resolved_data_dir().join("claude-config")
	}

	pub fn resolved_working_dir(&self) -> PathBuf {
		self.working_dir.clone().unwrap_or_else(|| {
			if self.no_isolate {
				self.resolved_data_dir()
			} else {
				self.isolated_config_dir()
			}
		})
	}

	pub fn stats_db_path(&self) -> PathBuf {
		self.resolved_data_dir().join("stats.redb")
	}

	/// Create a per-request config dir with a fresh credentials symlink.
	/// Returns the path on success, or `None` if isolation is disabled or
	/// setup fails. The caller must remove the directory when done.
	pub fn create_request_config_dir(&self, request_id: &str) -> Option<PathBuf> {
		if self.no_isolate {
			return None;
		}
		let dir = self.isolated_config_dir().join(request_id);
		if let Err(e) = std::fs::create_dir_all(&dir) {
			tracing::warn!(path = ?dir, error = %e, "Failed to create request config dir");
			return None;
		}
		let creds_source = self.credentials_source();
		let creds_dest = dir.join(CREDENTIALS_FILENAME);
		#[cfg(unix)]
		if let Err(e) = std::os::unix::fs::symlink(&creds_source, &creds_dest) {
			tracing::warn!(error = %e, "Failed to symlink credentials for request");
			let _ = std::fs::remove_dir_all(&dir);
			return None;
		}
		Some(dir)
	}

	/// Path to the host credentials file.
	pub fn credentials_source(&self) -> PathBuf {
		dirs::home_dir()
			.expect("Could not determine home directory")
			.join(".claude")
			.join(CREDENTIALS_FILENAME)
	}

	/// Load all API keys from --api-keys and --api-keys-file, deduplicated.
	pub fn load_api_keys(&self) -> Vec<String> {
		let mut keys: Vec<String> = self
			.api_keys
			.iter()
			.map(|k| k.trim().to_string())
			.filter(|k| !k.is_empty())
			.collect();

		if let Some(ref path) = self.api_keys_file {
			if let Ok(contents) = std::fs::read_to_string(path) {
				for line in contents.lines() {
					let trimmed = line.trim();
					if !trimmed.is_empty() && !trimmed.starts_with('#') {
						keys.push(trimmed.to_string());
					}
				}
			} else {
				tracing::error!("Failed to read API keys file: {:?}", path);
				std::process::exit(1);
			}
		}

		keys.sort();
		keys.dedup();
		keys
	}
}
