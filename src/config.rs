use std::path::PathBuf;

use clap::Parser;

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

	/// Data directory for the stats DB.
	#[arg(long, env = "CCP_DATA_DIR")]
	pub data_dir: Option<PathBuf>,

	/// API keys (comma-separated). If set, requests must include a valid key.
	#[arg(long, env = "CCP_API_KEYS", value_delimiter = ',')]
	pub api_keys: Vec<String>,

	/// File containing API keys (one per line, # comments allowed).
	#[arg(long, env = "CCP_API_KEYS_FILE")]
	pub api_keys_file: Option<PathBuf>,

	/// Disable authentication entirely.
	#[arg(long, env = "CCP_NO_AUTH")]
	pub no_auth: bool,

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
