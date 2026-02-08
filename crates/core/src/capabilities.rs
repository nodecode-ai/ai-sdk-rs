use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::{OnceLock, RwLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub reasoning: Option<bool>,
    #[serde(default)]
    pub temperature: Option<bool>,
    // Placeholder for future endpoint hints, etc.
    #[serde(default)]
    pub supports_responses_api: Option<bool>,
}

#[derive(Debug)]
struct CacheEntry {
    value: Value,
    fetched_at: Instant,
}

static CAPS: OnceLock<RwLock<Option<CacheEntry>>> = OnceLock::new();

fn ttl() -> Duration {
    Duration::from_secs(300)
}

fn default_index_path() -> PathBuf {
    // Default to clixode providers cache path
    if let Some(proj) = directories::ProjectDirs::from("com", "clixode", "clixode") {
        return proj.cache_dir().join("api").join("api.json");
    }
    // Fallback to home dir if dirs not available (unlikely)
    let mut p = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    p.push(".cache/clixode/api/api.json");
    p
}

fn read_index_from_env() -> Option<Value> {
    if let Ok(inline) = std::env::var("AI_SDK_PROVIDERS_INDEX_JSON") {
        if !inline.trim().is_empty() {
            if let Ok(v) = serde_json::from_str::<Value>(&inline) {
                return Some(v);
            }
        }
    }
    if let Ok(path) = std::env::var("AI_SDK_PROVIDERS_INDEX_PATH") {
        let pb = PathBuf::from(path);
        if let Ok(bytes) = std::fs::read(pb) {
            if let Ok(v) = serde_json::from_slice::<Value>(&bytes) {
                return Some(v);
            }
        }
    }
    None
}

fn read_index_from_disk() -> Option<Value> {
    if let Ok(disable) = std::env::var("AI_SDK_CAPS_DISABLE_DISK") {
        if disable == "1" || disable.eq_ignore_ascii_case("true") {
            return None;
        }
    }
    let p = default_index_path();
    if let Ok(bytes) = std::fs::read(p) {
        serde_json::from_slice::<Value>(&bytes).ok()
    } else {
        None
    }
}

fn load_index() -> Option<Value> {
    let lock = CAPS.get_or_init(|| RwLock::new(None));
    // memory cache
    if let Ok(c) = lock.read() {
        if let Some(ref e) = *c {
            if e.fetched_at.elapsed() < ttl() {
                return Some(e.value.clone());
            }
        }
    }
    // env first, then disk
    let val = read_index_from_env().or_else(read_index_from_disk)?;
    if let Ok(mut c) = lock.write() {
        *c = Some(CacheEntry {
            value: val.clone(),
            fetched_at: Instant::now(),
        });
    }
    Some(val)
}

/// Return selected model capabilities from the providers index, if available.
pub fn get_model_capabilities(provider: &str, model: &str) -> Option<ModelCapabilities> {
    let val = load_index()?;
    let providers = val.get("providers")?.as_array()?;
    let pl = provider.to_ascii_lowercase();
    let mnorm = model.split_once('/').map(|(_, tail)| tail).unwrap_or(model);
    for p in providers {
        let id = p.get("id").and_then(|v| v.as_str());
        let pr = p.get("provider").and_then(|v| v.as_str());
        let matches = id.map(|s| s.eq_ignore_ascii_case(&pl)).unwrap_or(false)
            || pr.map(|s| s.eq_ignore_ascii_case(&pl)).unwrap_or(false);
        if !matches {
            continue;
        }
        if let Some(models) = p.get("models").and_then(|v| v.as_array()) {
            for m in models {
                if m.get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.eq_ignore_ascii_case(mnorm))
                    .unwrap_or(false)
                {
                    let caps = m.get("capabilities");
                    let reasoning = caps
                        .and_then(|c| c.get("reasoning"))
                        .and_then(|v| v.as_bool());
                    let temperature = caps
                        .and_then(|c| c.get("temperature"))
                        .and_then(|v| v.as_bool());
                    // Tolerant detection of responses support from a few likely shapes
                    let mut supports_responses_api = caps
                        .and_then(|c| c.get("responses_api"))
                        .and_then(|v| v.as_bool());
                    if supports_responses_api.is_none() {
                        supports_responses_api = caps
                            .and_then(|c| c.get("supports_responses_api"))
                            .and_then(|v| v.as_bool());
                    }
                    if supports_responses_api.is_none() {
                        supports_responses_api = caps
                            .and_then(|c| c.get("responses"))
                            .and_then(|v| v.as_bool());
                    }
                    if supports_responses_api.is_none() {
                        // endpoints: { responses: true } OR endpoints: ["responses", ...]
                        if let Some(ep) = m.get("endpoints") {
                            if let Some(b) = ep.get("responses").and_then(|v| v.as_bool()) {
                                supports_responses_api = Some(b);
                            } else if let Some(arr) = ep.as_array() {
                                let present = arr.iter().any(|v| {
                                    v.as_str()
                                        .map(|s| s.eq_ignore_ascii_case("responses"))
                                        .unwrap_or(false)
                                });
                                if present {
                                    supports_responses_api = Some(true);
                                }
                            }
                        }
                    }
                    return Some(ModelCapabilities {
                        reasoning,
                        temperature,
                        supports_responses_api,
                    });
                }
            }
        }
    }
    None
}
