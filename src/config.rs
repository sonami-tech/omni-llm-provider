use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Clone, Debug)]
#[command(
	name = "claude-code-provider",
	version,
	about = "OpenAI-compatible API proxy backed by Claude Code CLI"
)]
pub struct Config {
	/// Listen port.
	#[arg(short = 'p', long, default_value = "18321", env = "CCP_PORT")]
	pub port: u16,

	/// Listen address.
	#[arg(short = 'H', long, default_value = "127.0.0.1", env = "CCP_HOST")]
	pub host: String,

	/// Max concurrent subprocesses.
	#[arg(short = 'c', long, default_value = "5", env = "CCP_MAX_CONCURRENT")]
	pub max_concurrent: usize,

	/// Subprocess inactivity timeout in seconds.
	#[arg(short = 't', long, default_value = "600", env = "CCP_TIMEOUT")]
	pub timeout: u64,

	/// Max time a request waits in queue (seconds).
	#[arg(short = 'q', long, default_value = "60", env = "CCP_QUEUE_TIMEOUT")]
	pub queue_timeout: u64,

	/// Max agentic turns per request (passed to CLI as --max-turns).
	#[arg(long, default_value = "3", env = "CCP_MAX_TURNS")]
	pub max_turns: u32,

	/// Path to claude CLI binary.
	#[arg(long, default_value = "claude", env = "CCP_CLAUDE_PATH")]
	pub claude_path: String,

	/// Data directory for config isolation and stats DB.
	#[arg(long, env = "CCP_DATA_DIR")]
	pub data_dir: Option<PathBuf>,

	/// Working directory for subprocesses (defaults to isolated config dir).
	#[arg(long, env = "CCP_WORKING_DIR")]
	pub working_dir: Option<PathBuf>,

	/// Disable config isolation (use host's ~/.claude directly).
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
