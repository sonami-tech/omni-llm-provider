mod auth;
mod config;
mod conversation_log;
mod error;
mod log_color;
mod models;
mod replacements;
mod routes;
mod session;
mod stats;
mod time_util;
mod translate;
mod upstream;

pub const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;
const MIN_API_KEY_LENGTH: usize = 8;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use clap::Parser;
use tokio::net::TcpListener;
use tracing::{error, info, warn};

use config::Config;
use error::AppError;
use upstream::credentials::Credentials;

pub struct AppState {
	pub config: Config,
	pub stats: Arc<stats::Stats>,
	pub conversation_log: Option<Arc<conversation_log::ConversationLog>>,
	pub replacements: Arc<replacements::Replacements>,
	pub upstream: upstream::UpstreamClient,
}

#[tokio::main]
async fn main() {
	let config = Config::parse();

	let filter = if config.verbose {
		"claude_code_provider=debug"
	} else {
		"claude_code_provider=info"
	};
	let color_mode = log_color::ColorMode::from_env();
	tracing_subscriber::fmt()
		.with_env_filter(
			tracing_subscriber::EnvFilter::try_from_default_env()
				.unwrap_or_else(|_| filter.parse().unwrap()),
		)
		.with_target(false)
		.with_writer(std::io::stderr)
		.with_ansi(matches!(color_mode, log_color::ColorMode::On))
		.fmt_fields(log_color::ColorFields::new(color_mode))
		.compact()
		.init();

	// ── Startup validation ────────────────────────────────────

	let creds_path = Credentials::default_path();
	let creds = Credentials::load_fresh(&creds_path).unwrap_or_else(|e| {
		error!(
			"Claude OAuth credentials are not ready at {:?}: {}. Run 'claude login' first.",
			creds_path, e
		);
		std::process::exit(1);
	});
	if let Err(e) = creds.check_expired() {
		error!(
			"Claude OAuth credentials at {:?} are expired: {}. Run 'claude' once to refresh.",
			creds_path, e
		);
		std::process::exit(1);
	}
	if let Some(sub_type) = creds.subscription_type.as_deref() {
		info!("Authenticated as {} subscriber", sub_type);
	} else {
		info!("Authenticated with Claude OAuth credentials");
	}

	info!("Stats DB: {:?}", config.stats_db_path());

	// ── Server setup ──────────────────────────────────────────

	let stats_db = stats::Stats::open(config.stats_db_path()).unwrap_or_else(|e| {
		error!("Failed to open stats database: {}", e);
		std::process::exit(1);
	});

	// Setup conversation logging.
	let log_conversations = config.log_conversations || config.log_file.is_some();
	let conversation_log = if log_conversations {
		let log = if let Some(ref path) = config.log_file {
			conversation_log::ConversationLog::to_file(path).unwrap_or_else(|e| {
				error!("Failed to open log file {:?}: {}", path, e);
				std::process::exit(1);
			})
		} else {
			conversation_log::ConversationLog::to_stderr()
		};
		if let Some(ref path) = config.log_file {
			info!("Conversation logging to file: {:?}", path);
		} else {
			info!("Conversation logging to stderr");
		}
		Some(Arc::new(log))
	} else {
		None
	};

	// Load replacement rules.
	let replacements = if let Some(ref path) = config.replace_rules {
		let r = replacements::Replacements::load(path).unwrap_or_else(|e| {
			error!("{}", e);
			std::process::exit(1);
		});
		info!("Loaded {} replacement rules from {:?}", r.count(), path);
		Arc::new(r)
	} else {
		Arc::new(replacements::Replacements::empty())
	};

	let upstream_client = match upstream::UpstreamClient::new() {
		Ok(c) => c,
		Err(e) => {
			error!("Failed to build upstream HTTPS client: {e}");
			std::process::exit(1);
		}
	};

	let state = Arc::new(AppState {
		config: config.clone(),
		stats: Arc::new(stats_db),
		conversation_log,
		replacements,
		upstream: upstream_client,
	});

	// Resolve API keys: no-auth, explicit, or auto-generated.
	let api_keys: HashSet<String> = if config.no_auth {
		warn!("Authentication disabled. All endpoints are open.");
		HashSet::new()
	} else {
		let explicit = config.load_api_keys();
		if explicit.is_empty() {
			// Auto-generate a key.
			let key = uuid::Uuid::new_v4().simple().to_string();
			info!("Auto-generated API key: {}", key);
			HashSet::from([key])
		} else {
			// Validate minimum length.
			for key in &explicit {
				if key.len() < MIN_API_KEY_LENGTH {
					error!("API key too short (minimum {} characters): \"{}...\"", MIN_API_KEY_LENGTH, &key[..key.len().min(4)]);
					std::process::exit(1);
				}
			}
			let count = explicit.len();
			info!("API key auth enabled ({} key{})", count, if count == 1 { "" } else { "s" });
			explicit.into_iter().collect()
		}
	};
	let api_keys = Arc::new(api_keys);

	// API routes (auth-protected).
	let api_keys_clone = api_keys.clone();
	let api_routes = Router::new()
		.route("/v1/models", get(routes::models::models_handler))
		.route("/models", get(routes::models::models_handler))
		.route(
			"/v1/chat/completions",
			post(routes::completions::completions_handler),
		)
		.route(
			"/chat/completions",
			post(routes::completions::completions_handler),
		)
		.layer(middleware::from_fn(move |req, next| {
			auth::auth_layer(api_keys_clone.clone(), req, next)
		}));

	let app = Router::new()
		.merge(api_routes)
		.route("/health", get(routes::health::health_handler))
		.route("/stats", get(routes::stats::stats_html_handler))
		.route("/stats/json", get(routes::stats::stats_json_handler))
		.fallback(fallback_handler)
		.layer(tower_http::cors::CorsLayer::permissive())
		.layer(DefaultBodyLimit::max(MAX_BODY_SIZE))
		.with_state(state);

	let addr: SocketAddr = format!("{}:{}", config.host, config.port)
		.parse()
		.unwrap_or_else(|e| {
			error!("Invalid listen address: {}", e);
			std::process::exit(1);
		});

	let listener = match TcpListener::bind(addr).await {
		Ok(l) => l,
		Err(e) => {
			error!("Failed to bind to {}: {}", addr, e);
			if e.kind() == std::io::ErrorKind::AddrInUse {
				error!("Port {} is already in use", config.port);
			}
			std::process::exit(1);
		}
	};

	let display_host = if config.host == "0.0.0.0" {
		detect_lan_ip().unwrap_or_else(|| "127.0.0.1".to_string())
	} else {
		config.host.clone()
	};
	info!(
		"Claude Code Provider v{} listening on http://{}:{}",
		env!("CARGO_PKG_VERSION"),
		display_host,
		config.port,
	);
	info!("Using direct Anthropic Messages upstream");

	let shutdown = async {
		let ctrl_c = tokio::signal::ctrl_c();
		#[cfg(unix)]
		{
			let mut sigterm =
				tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
					.expect("Failed to install SIGTERM handler");
			tokio::select! {
				_ = ctrl_c => { info!("Received SIGINT, shutting down..."); }
				_ = sigterm.recv() => { info!("Received SIGTERM, shutting down..."); }
			}
		}
		#[cfg(not(unix))]
		{
			ctrl_c.await.ok();
			info!("Received SIGINT, shutting down...");
		}
	};

	axum::serve(listener, app)
		.with_graceful_shutdown(shutdown)
		.await
		.unwrap_or_else(|e| {
			error!("Server error: {}", e);
			std::process::exit(1);
		});

	info!("Server stopped.");
}

async fn fallback_handler() -> AppError {
	AppError::NotFound("The requested endpoint does not exist".into())
}

/// Detect LAN IP by asking the OS which interface routes to the internet.
/// Does not send any traffic.
fn detect_lan_ip() -> Option<String> {
	let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
	socket.connect("8.8.8.8:80").ok()?;
	let addr = socket.local_addr().ok()?;
	Some(addr.ip().to_string())
}
