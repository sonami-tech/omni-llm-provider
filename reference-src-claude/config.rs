use std::path::PathBuf;

use clap::Parser;

use crate::conversation_log::{DEFAULT_LOG_BACKUPS, DEFAULT_LOG_MAX_BYTES};

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
    #[arg(long, env = "CCP_LOG_FILE", conflicts_with = "log_dir")]
    pub log_file: Option<PathBuf>,

    /// Directory to write one conversation log file per session id.
    #[arg(long, env = "CCP_LOG_DIR", conflicts_with = "log_file")]
    pub log_dir: Option<PathBuf>,

    /// Rotate --log-file after this many bytes. Set to 0 to disable rotation.
    #[arg(long, env = "CCP_LOG_MAX_BYTES", default_value_t = DEFAULT_LOG_MAX_BYTES)]
    pub log_max_bytes: u64,

    /// Number of rotated conversation log files to keep.
    #[arg(long, env = "CCP_LOG_BACKUPS", default_value_t = DEFAULT_LOG_BACKUPS)]
    pub log_backups: usize,

    /// Skip prepending Claude Code identity blocks to outbound requests.
    /// This disables both the billing marker and canonical Claude Code
    /// preamble. The identity blocks are required for opus/sonnet calls to
    /// bypass Anthropic's OAuth subscription gate; disable only if the
    /// consumer is providing equivalent identity blocks or for upstream
    /// debugging.
    #[arg(long, env = "CCP_NO_PREAMBLE")]
    pub no_preamble: bool,

    /// Claude Code fingerprint profile to claim upstream. Use "latest" to
    /// track the newest known-good profile, or pin a concrete profile name.
    #[arg(long, env = "CCP_FINGERPRINT_PROFILE", default_value = "latest")]
    pub fingerprint_profile: String,

    /// Enable debug logging.
    #[arg(short = 'v', long)]
    pub verbose: bool,
}

impl Config {
    pub fn resolved_data_dir(&self) -> PathBuf {
        self.data_dir.clone().unwrap_or_else(|| {
            // Fall back to a relative directory rather than panicking when the
            // OS data dir can't be determined; `--data-dir` / CCP_DATA_DIR
            // overrides this entirely.
            dirs::data_dir().map_or_else(
                || {
                    tracing::warn!(
                        "Could not determine OS data directory; using ./claude-code-provider-data. Set --data-dir to override."
                    );
                    PathBuf::from("claude-code-provider-data")
                },
                |d| d.join("claude-code-provider"),
            )
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn conversation_log_rotation_defaults_are_set() {
        let config = Config::try_parse_from(["ccp"]).unwrap();

        assert_eq!(config.log_max_bytes, DEFAULT_LOG_MAX_BYTES);
        assert_eq!(config.log_backups, DEFAULT_LOG_BACKUPS);
    }

    #[test]
    fn conversation_log_rotation_options_parse() {
        let config = Config::try_parse_from([
            "ccp",
            "--log-max-bytes",
            "1024",
            "--log-backups",
            "2",
        ])
        .unwrap();

        assert_eq!(config.log_max_bytes, 1024);
        assert_eq!(config.log_backups, 2);
    }

    #[test]
    fn conversation_log_dir_option_parse() {
        let config = Config::try_parse_from(["ccp", "--log-dir", "/tmp/ccp-logs"]).unwrap();

        assert_eq!(config.log_dir, Some(PathBuf::from("/tmp/ccp-logs")));
    }

    #[test]
    fn conversation_log_dir_conflicts_with_log_file() {
        let result = Config::try_parse_from([
            "ccp",
            "--log-file",
            "/tmp/ccp.log",
            "--log-dir",
            "/tmp/ccp-logs",
        ]);

        assert!(result.is_err());
    }
}
