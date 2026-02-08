use ai_sdk_rs::core::transport::TransportConfig;
use ai_sdk_rs::core::{EmbedOptions, EmbeddingModel};
use ai_sdk_rs::providers::openai_compatible::embedding::embedding_model::OpenAICompatibleEmbeddingConfig;
use ai_sdk_rs::providers::openai_compatible::OpenAICompatibleEmbeddingModel;
use anyhow::Result;

// Run with:
//   CHUTES_API_TOKEN=... cargo run -p embed-openai-compatible -- "example-string"
// Optional:
//   EMBED_BASE_URL=https://chutes-qwen-qwen3-embedding-0-6b.chutes.ai/v1 (default)
//   EMBED_MODEL=qwen3-embedding-0-6b (default)
//   EMBED_INPUT=example-string (if no CLI arg is provided)

fn strip_prefix(value: String, prefix: &str) -> String {
    value
        .strip_prefix(prefix)
        .map(|s| s.to_string())
        .unwrap_or(value)
}

#[tokio::main]
async fn main() -> Result<()> {
    let input = std::env::args()
        .nth(1)
        .map(|v| strip_prefix(v, "input="))
        .or_else(|| {
            std::env::var("EMBED_INPUT")
                .ok()
                .map(|v| strip_prefix(v, "input="))
        })
        .unwrap_or_else(|| "example-string".to_string());
    let model = std::env::args()
        .nth(2)
        .map(|v| strip_prefix(v, "model="))
        .or_else(|| {
            std::env::var("EMBED_MODEL")
                .ok()
                .map(|v| strip_prefix(v, "model="))
        })
        .unwrap_or_else(|| "qwen3-embedding-0-6b".to_string());
    let base_url = std::env::var("EMBED_BASE_URL")
        .unwrap_or_else(|_| "https://chutes-qwen-qwen3-embedding-0-6b.chutes.ai/v1".to_string());

    let api_key = std::env::var("CHUTES_API_TOKEN").unwrap_or_default();
    if api_key.is_empty() {
        eprintln!("Set CHUTES_API_TOKEN before running.");
        std::process::exit(1);
    }

    let cfg = OpenAICompatibleEmbeddingConfig {
        provider_scope_name: "openai-compatible".to_string(),
        base_url,
        headers: vec![("authorization".into(), format!("Bearer {}", api_key))],
        http: ai_sdk_rs::transports::reqwest::ReqwestTransport::default(),
        transport_cfg: TransportConfig::default(),
        query_params: vec![],
        max_embeddings_per_call: None,
        supports_parallel_calls: true,
        default_options: None,
    };

    let provider = OpenAICompatibleEmbeddingModel::new(model, cfg);
    let options = EmbedOptions::new(vec![input]);

    let resp = provider.do_embed(options).await?;

    println!("Embeddings returned: {}", resp.embeddings.len());
    if let Some(first) = resp.embeddings.first() {
        let head = &first[..first.len().min(8)];
        println!("First embedding length: {}, head: {:?}", first.len(), head);
    }
    if let Some(usage) = resp.usage {
        if let Some(tokens) = usage.tokens {
            println!("Prompt tokens: {}", tokens);
        }
    }

    Ok(())
}
