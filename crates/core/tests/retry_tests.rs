use crate::ai_sdk_core::retry::RetryConfig;
use std::time::Duration;

#[test]
fn test_backoff_calculation() {
    let config = RetryConfig::custom(5, Duration::from_millis(100), Duration::from_secs(10));

    assert_eq!(
        config.calculate_backoff(1, None),
        Duration::from_millis(100)
    );
    assert_eq!(
        config.calculate_backoff(2, None),
        Duration::from_millis(200)
    );
    assert_eq!(
        config.calculate_backoff(3, None),
        Duration::from_millis(400)
    );
    assert_eq!(
        config.calculate_backoff(4, None),
        Duration::from_millis(800)
    );
    assert_eq!(
        config.calculate_backoff(5, None),
        Duration::from_millis(1600)
    );

    assert_eq!(config.calculate_backoff(10, None), Duration::from_secs(10));

    assert_eq!(
        config.calculate_backoff(1, Some(Duration::from_secs(5))),
        Duration::from_secs(5)
    );
    assert_eq!(
        config.calculate_backoff(1, Some(Duration::from_secs(20))),
        Duration::from_secs(10)
    );
}

#[test]
fn test_presets() {
    let quick = RetryConfig::quick();
    assert_eq!(quick.max_retries, 3);
    assert_eq!(quick.initial_interval, Duration::from_millis(50));

    let network = RetryConfig::network();
    assert_eq!(network.max_retries, 5);
    assert_eq!(network.initial_interval, Duration::from_millis(250));

    let api = RetryConfig::api();
    assert_eq!(api.max_retries, 5);
    assert_eq!(api.initial_interval, Duration::from_secs(1));
}
