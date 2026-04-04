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
	#[arg(short = 'p', long, default_value = "3456", env = "CCP_PORT")]
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

	/// Path to claude CLI binary.
	#[arg(long, default_value = "claude", env = "CCP_CLAUDE_PATH")]
	pub claude_path: String,

	/// Data directory for config isolation and stats DB.
	#[arg(long, env = "CCP_DATA_DIR")]
	pub data_dir: Option<PathBuf>,

	/// Working directory for subprocesses (defaults to isolated config dir).
	#[arg(long, env = "CCP_WORKING_DIR")]
	pub working_dir: Option<PathBuf>,

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
		self.working_dir
			.clone()
			.unwrap_or_else(|| self.isolated_config_dir())
	}

	pub fn stats_db_path(&self) -> PathBuf {
		self.resolved_data_dir().join("stats.redb")
	}
}
