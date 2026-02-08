# ai-sdk-rs (v0.1.1)

A tiny, open-source Rust AI SDK inspired by [Vercel's AI SDK](https://github.com/vercel/ai), but in Rust.

## Features (v0.1.1)
- Unified `LanguageModel` trait + common chat types
- `EmbeddingModel` trait + OpenAI-compatible embeddings endpoint
- Built-in providers: OpenAI, Azure OpenAI, Anthropic, Google, Google Vertex, Amazon Bedrock, Gateway
- OpenAI-compatible adapter modes for third-party compatible endpoints
- Google (Gemini / AI Studio) provider
- Streaming to a normalized `Event` stream
- Examples: text and streaming CLI

## Supported Providers (Built-in)

- `openai`
- `azure`
- `anthropic`
- `google`
- `google-vertex`
- `amazon-bedrock`
- `gateway`

Compatibility adapters (not part of the built-in list): `openai-compatible`, `openai-compatible-chat`, `openai-compatible-completion`.

## Installation

```bash
cargo add ai-sdk-rs
```

## Quick start

```bash
# 1) clone files, then:
cd ai-sdk-rs

# 2) set your key (or point to a local OpenAI-compatible server via base_url)
export OPENAI_API_KEY=sk-...

# 3) run the examples
# Text (one-shot):
cargo run -p generate-text

# Streaming (SSE-style event stream mapped to stdout/stderr):
cargo run -p generate-stream

# 4) Optionally try Google AI Studio (Gemini)
# export GEMINI_API_KEY=...
# export GEMINI_MODEL=gemini-2.5-flash
# (then adapt example code to use the Google provider)
```

## Anthropic: minimal streaming example

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# optional:
# Sonnet 4.5 (default below):
export MODEL=claude-sonnet-4-5-20250929
# Or Opus 4.6:
# export MODEL=claude-opus-4-6
```

`Cargo.toml` (minimal deps for this snippet):

```toml
[dependencies]
ai-sdk-rs = "0.1.1"
futures-util = "0.3"
serde_json = "1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use ai_sdk_rs::core::types as v2t;
use ai_sdk_rs::provider::{registry, Credentials};
use ai_sdk_rs::providers::anthropic as _;
use ai_sdk_rs::types::catalog::{ProviderDefinition, SdkType};
use futures_util::StreamExt;
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let model = std::env::var("MODEL").unwrap_or_else(|_| "claude-sonnet-4-5-20250929".into());

    let reg = registry::iter()
        .into_iter()
        .find(|entry| entry.id.eq_ignore_ascii_case("anthropic"))
        .expect("anthropic provider registration not found");

    let def = ProviderDefinition {
        name: "anthropic".into(),
        display_name: "Anthropic".into(),
        sdk_type: SdkType::Anthropic,
        base_url: "https://api.anthropic.com/v1".into(),
        env: Some("ANTHROPIC_API_KEY".into()),
        npm: None,
        doc: None,
        endpoint_path: "".into(),
        headers: HashMap::new(),
        query_params: HashMap::new(),
        stream_idle_timeout_ms: None,
        auth_type: "api-key".into(),
        models: HashMap::new(),
        preserve_model_prefix: true,
    };

    let lm = (reg.build)(&def, &model, &Credentials::ApiKey(api_key))?;

    let mut options = v2t::CallOptions::new(vec![
        v2t::PromptMessage::System {
            content: "You are a concise assistant.".into(),
            provider_options: None,
        },
        v2t::PromptMessage::User {
            content: vec![v2t::UserPart::Text {
                text: "Briefly explain how rainbows form.".into(),
                provider_options: None,
            }],
            provider_options: None,
        },
    ]);
    // Anthropic thinking requires max_output_tokens > budget_tokens.
    options.max_output_tokens = Some(1200);
    options.provider_options.insert(
        "anthropic".into(),
        HashMap::from([(
            "thinking".into(),
            json!({ "type": "enabled", "budget_tokens": 1024 }),
        )]),
    );

    let mut stream = lm.do_stream(options).await?.stream;
    while let Some(part) = stream.next().await {
        match part? {
            v2t::StreamPart::ReasoningStart { .. } => eprintln!("<thinking>"),
            v2t::StreamPart::ReasoningDelta { delta, .. } => eprint!("{}", delta),
            v2t::StreamPart::ReasoningEnd { .. } => eprintln!("\n</thinking>"),
            v2t::StreamPart::TextDelta { delta, .. } => print!("{}", delta),
            v2t::StreamPart::Error { error } => eprintln!("\n[error] {}", error),
            v2t::StreamPart::Finish { .. } => break,
            _ => {}
        }
    }

    Ok(())
}
```

## License

MIT License

Copyright (c) 2026 Nodecode.ai

See [`LICENSE`](LICENSE) for the full text.
