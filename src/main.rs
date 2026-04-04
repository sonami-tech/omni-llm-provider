mod config;
mod error;
mod models;
mod routes;
mod stats;
mod subprocess;
mod translate;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use clap::Parser;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tracing::{error, info, warn};

use config::Config;
use error::AppError;

pub struct AppState {
	pub config: Config,
	pub semaphore: Arc<Semaphore>,
	pub stats: Arc<stats::Stats>,
}

#[tokio::main]
async fn main() {
	let config = Config::parse();

	let filter = if config.verbose {
		"claude_code_provider=debug"
	} else {
		"claude_code_provider=info"
	};
	tracing_subscriber::fmt()
		.with_env_filter(
			tracing_subscriber::EnvFilter::try_from_default_env()
				.unwrap_or_else(|_| filter.parse().unwrap()),
		)
		.with_target(false)
		.with_writer(std::io::stderr)
		.compact()
		.init();

	// ── Startup validation ────────────────────────────────────

	// 1. Verify claude binary exists.
	match tokio::process::Command::new(&config.claude_path)
		.arg("--version")
		.output()
		.await
	{
		Ok(output) => {
			let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
			info!("Found claude CLI: {}", version);
		}
		Err(e) => {
			error!(
				"claude CLI not found at '{}': {}. Install with: npm install -g @anthropic-ai/claude-code",
				config.claude_path, e
			);
			std::process::exit(1);
		}
	}

	// 2. Check authentication status (best-effort).
	match tokio::process::Command::new(&config.claude_path)
		.args(["auth", "status"])
		.output()
		.await
	{
		Ok(output) => {
			let stdout = String::from_utf8_lossy(&output.stdout);
			match serde_json::from_str::<serde_json::Value>(stdout.trim()) {
				Ok(json) => {
					if json.get("loggedIn") == Some(&serde_json::Value::Bool(true)) {
						if let Some(sub_type) = json.get("subscriptionType").and_then(|v| v.as_str())
						{
							info!("Authenticated as {} subscriber", sub_type);
						} else {
							info!("Authenticated (subscription type unknown)");
						}
					} else {
						error!("Claude Code is not logged in. Run 'claude login' first.");
						std::process::exit(1);
					}
				}
				Err(e) => {
					warn!("Could not parse auth status ({}), continuing anyway", e);
				}
			}
		}
		Err(e) => {
			warn!("Could not check auth status ({}), continuing anyway", e);
		}
	}

	// 3. Setup config directory.
	if config.no_isolate {
		info!("Config isolation disabled, using host Claude configuration");
	} else {
		let config_dir = config.isolated_config_dir();
		std::fs::create_dir_all(&config_dir).unwrap_or_else(|e| {
			error!("Failed to create config dir {:?}: {}", config_dir, e);
			std::process::exit(1);
		});

		// Clean all contents (stale .claude.json can enable unexpected tools).
		if let Ok(entries) = std::fs::read_dir(&config_dir) {
			for entry in entries.flatten() {
				let path = entry.path();
				if path.is_dir() {
					let _ = std::fs::remove_dir_all(&path);
				} else {
					let _ = std::fs::remove_file(&path);
				}
			}
		}

		// Create .credentials.json symlink.
		let home_dir = dirs::home_dir().expect("Could not determine home directory");
		let creds_source = home_dir.join(".claude").join(".credentials.json");
		let creds_dest = config_dir.join(".credentials.json");

		if !creds_source.exists() {
			error!(
				"Claude Code credentials not found at {:?}. Run 'claude login' first.",
				creds_source
			);
			std::process::exit(1);
		}

		#[cfg(unix)]
		{
			std::os::unix::fs::symlink(&creds_source, &creds_dest).unwrap_or_else(|e| {
				error!(
					"Failed to symlink {:?} -> {:?}: {}",
					creds_dest, creds_source, e
				);
				std::process::exit(1);
			});
		}

		info!("Isolated config dir: {:?}", config_dir);
	}

	// Ensure working directory exists.
	let working_dir = config.resolved_working_dir();
	std::fs::create_dir_all(&working_dir).unwrap_or_else(|e| {
		error!("Failed to create working dir {:?}: {}", working_dir, e);
		std::process::exit(1);
	});

	info!("Working dir: {:?}", working_dir);
	info!("Stats DB: {:?}", config.stats_db_path());

	// ── Server setup ──────────────────────────────────────────

	let semaphore = Arc::new(Semaphore::new(config.max_concurrent));

	let stats_db = stats::Stats::open(config.stats_db_path()).unwrap_or_else(|e| {
		error!("Failed to open stats database: {}", e);
		std::process::exit(1);
	});

	let state = Arc::new(AppState {
		config: config.clone(),
		semaphore,
		stats: Arc::new(stats_db),
	});

	let app = Router::new()
		.route("/health", get(routes::health::health_handler))
		.route("/v1/models", get(routes::models::models_handler))
		.route(
			"/v1/chat/completions",
			post(routes::completions::completions_handler),
		)
		.route("/stats", get(routes::stats::stats_html_handler))
		.route("/stats/json", get(routes::stats::stats_json_handler))
		.fallback(fallback_handler)
		.layer(tower_http::cors::CorsLayer::permissive())
		.layer(DefaultBodyLimit::max(10 * 1024 * 1024))
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

	info!(
		"Claude Code Provider v{} listening on http://{}",
		env!("CARGO_PKG_VERSION"),
		addr
	);
	info!(
		"max_concurrent={} timeout={}s queue_timeout={}s",
		config.max_concurrent, config.timeout, config.queue_timeout
	);

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
