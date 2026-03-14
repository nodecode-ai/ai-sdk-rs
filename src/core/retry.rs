//! Unified retry utilities for the SDK and external consumers
//!
//! This module provides a flexible retry system with exponential backoff,
//! configurable presets, and a trait-based approach for determining retryable errors.

use std::future::Future;
use std::time::Duration;

/// Configuration for retry behavior with exponential backoff
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (not including the initial attempt)
    pub max_retries: u32,
    /// Initial backoff delay
    pub initial_interval: Duration,
    /// Maximum backoff delay
    pub max_interval: Duration,
    /// Multiplier for exponential backoff (typically 2.0)
    pub multiplier: f64,
}

impl RetryConfig {
    /// Quick retry for fast operations (50ms initial, 1s max, 3 retries)
    pub fn quick() -> Self {
        Self {
            max_retries: 3,
            initial_interval: Duration::from_millis(50),
            max_interval: Duration::from_secs(1),
            multiplier: 2.0,
        }
    }

    /// Network retry for typical API calls (250ms initial, 10s max, 5 retries)
    pub fn network() -> Self {
        Self {
            max_retries: 5,
            initial_interval: Duration::from_millis(250),
            max_interval: Duration::from_secs(10),
            multiplier: 2.0,
        }
    }

    /// API retry for rate-limited operations (1s initial, 60s max, 5 retries)
    pub fn api() -> Self {
        Self {
            max_retries: 5,
            initial_interval: Duration::from_secs(1),
            max_interval: Duration::from_secs(60),
            multiplier: 2.0,
        }
    }

    /// Create a custom configuration
    pub fn custom(max_retries: u32, initial_interval: Duration, max_interval: Duration) -> Self {
        Self {
            max_retries,
            initial_interval,
            max_interval,
            multiplier: 2.0,
        }
    }

    /// Calculate the backoff duration for a given attempt
    ///
    /// # Arguments
    /// * `attempt` - The attempt number (1-based)
    /// * `retry_after` - Optional retry-after hint from the server
    pub fn calculate_backoff(&self, attempt: u32, retry_after: Option<Duration>) -> Duration {
        // Use retry_after if provided and within bounds
        if let Some(duration) = retry_after {
            return duration.min(self.max_interval);
        }

        // Calculate exponential backoff
        let multiplier = self.multiplier.powi(attempt.saturating_sub(1) as i32);
        let backoff_ms = (self.initial_interval.as_millis() as f64 * multiplier) as u64;
        let backoff = Duration::from_millis(backoff_ms);

        // Cap at max_interval
        backoff.min(self.max_interval)
    }
}

/// Trait for determining if an error is retryable
pub trait Retryable {
    /// Check if the error should trigger a retry
    fn is_retryable(&self) -> bool;

    /// Extract retry-after hint if available (in milliseconds for compatibility)
    fn retry_after_ms(&self) -> Option<u64> {
        None
    }
}

// Optional implementations for common error types when feature is enabled
#[cfg(feature = "reqwest")]
impl Retryable for reqwest::Error {
    fn is_retryable(&self) -> bool {
        self.is_timeout()
            || self.is_connect()
            || self
                .status()
                .is_none_or(|s| s.is_server_error() || s.as_u16() == 429)
    }

    fn retry_after_ms(&self) -> Option<u64> {
        // Check for rate limit status
        if let Some(status) = self.status() {
            if status.as_u16() == 429 {
                // Try to extract retry-after from error details if available
                // For now, return None as reqwest doesn't expose headers from errors
                return None;
            }
        }
        None
    }
}

impl Retryable for std::io::Error {
    fn is_retryable(&self) -> bool {
        use std::io::ErrorKind::*;
        matches!(
            self.kind(),
            ConnectionRefused | ConnectionReset | ConnectionAborted | TimedOut | Interrupted
        )
    }
}

/// Execute an operation with retry logic
///
/// # Arguments
/// * `config` - Retry configuration
/// * `operation` - Async function that returns a Result
/// * `on_retry` - Optional callback for retry events
///
/// # Example
/// ```rust,no_run
/// use ai_sdk_rs::core::retry::{retry_with_backoff, RetryConfig};
///
/// async fn fetch_data() -> Result<String, std::io::Error> {
///     Ok("ok".to_string())
/// }
///
/// #[tokio::main(flavor = "current_thread")]
/// async fn main() {
/// let result = retry_with_backoff(
///     RetryConfig::network(),
///     || async { fetch_data().await },
///     |attempt, delay, error| {
///         println!("Retry {} after {:?}: {:?}", attempt, delay, error);
///     }
/// ).await;
///
/// let _ = result;
/// }
/// ```
pub async fn retry_with_backoff<F, Fut, T, E, R>(
    config: RetryConfig,
    mut operation: F,
    mut on_retry: R,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: Retryable + std::fmt::Debug,
    R: FnMut(u32, Duration, &E),
{
    let mut attempt = 0u32;

    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                // Check if error is retryable
                if !err.is_retryable() {
                    return Err(err);
                }

                // Check if we've exceeded max retries
                attempt += 1;
                if attempt > config.max_retries {
                    return Err(err);
                }

                // Calculate backoff with retry_after hint
                let retry_after = err.retry_after_ms().map(Duration::from_millis);
                let delay = config.calculate_backoff(attempt, retry_after);

                // Notify about retry
                on_retry(attempt, delay, &err);

                // Wait before retrying
                tokio::time::sleep(delay).await;
            }
        }
    }
}

/// Simplified retry function without callbacks
pub async fn retry<F, Fut, T, E>(config: RetryConfig, operation: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: Retryable + std::fmt::Debug,
{
    retry_with_backoff(config, operation, |_, _, _| {}).await
}
