use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::sync::Semaphore;

use crate::config::Config;
use crate::error::AppError;
use crate::subprocess::process::{SubprocessEvent, run_subprocess};

/// Acquire a concurrency slot, then spawn the subprocess in a background task.
/// The semaphore permit is held for the full duration of the subprocess.
pub async fn spawn_managed(
	config: Config,
	semaphore: Arc<Semaphore>,
	queue_timeout: Duration,
	request_id: String,
	cli_args: Vec<String>,
	tx: mpsc::Sender<SubprocessEvent>,
) -> Result<(), AppError> {
	let permit = match tokio::time::timeout(queue_timeout, semaphore.clone().acquire_owned()).await
	{
		Ok(Ok(permit)) => permit,
		Ok(Err(_)) => {
			return Err(AppError::ServerError("Semaphore closed".into()));
		}
		Err(_) => {
			return Err(AppError::ServiceUnavailable(format!(
				"All {} slots busy, timed out after {}s",
				config.max_concurrent,
				queue_timeout.as_secs()
			)));
		}
	};

	tokio::spawn(async move {
		let _permit = permit; // Held until task completes.
		run_subprocess(&config, &request_id, cli_args, tx).await;
	});

	Ok(())
}
