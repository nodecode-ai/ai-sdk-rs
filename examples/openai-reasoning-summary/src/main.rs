use ai_sdk_rs::core::types as v2t;
use ai_sdk_rs::core::LanguageModel;
use ai_sdk_rs::providers::openai::responses::language_model::OpenAIResponsesLanguageModel;
use anyhow::Result;
use futures_util::StreamExt;

// Run with:
//   OPENAI_API_KEY=sk-... OPENAI_MODEL=<reasoning-model> cargo run -p openai-reasoning-summary
// Notes:
// - Choose a reasoning-capable OPENAI_MODEL (e.g., o3, o4-mini, gpt-5 variants).
// - This example requests a reasoning summary via provider_overrides built
//   using the typed OpenAIOverridesBuilder.

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-5".to_string());
    let base_url = std::env::var("OPENAI_BASE_URL").ok();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    if api_key.is_empty() {
        eprintln!("OPENAI_API_KEY is not set. Set it in your environment.");
        std::process::exit(1);
    }

    let lm = OpenAIResponsesLanguageModel::create_simple(model, base_url, api_key);

    // Provider options: request reasoning summary and low effort
    let mut provider_options: v2t::ProviderOptions = std::collections::HashMap::new();
    provider_options.insert(
        "openai".into(),
        std::collections::HashMap::from([
            ("reasoningSummary".into(), serde_json::json!("auto")),
            ("reasoningEffort".into(), serde_json::json!("low")),
        ]),
    );

    let options = v2t::CallOptions {
        prompt: vec![
            v2t::PromptMessage::System {
                content: "You are a careful assistant. Think and then answer.".into(),
                provider_options: None,
            },
            v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "Just say. But think before saying that.".into(),
                    provider_options: None,
                }],
                provider_options: None,
            },
        ],
        max_output_tokens: Some(400),
        provider_options,
        ..Default::default()
    };

    let mut stream = lm.do_stream(options).await?.stream;

    let debug_events = std::env::var("AI_SDK_DEBUG_EVENTS")
        .ok()
        .filter(|v| v != "0")
        .is_some();
    let mut reasoning_started = false;
    while let Some(part) = stream.next().await {
        let ev = part?;
        if debug_events {
            if let Ok(line) = serde_json::to_string(&ev) {
                eprintln!("[event] {}", line);
            }
        }
        match ev {
            v2t::StreamPart::ReasoningStart { .. } => {
                if !reasoning_started {
                    reasoning_started = true;
                    eprintln!("<thinking>");
                }
            }
            v2t::StreamPart::ReasoningDelta { delta, .. } => {
                eprint!("{}", delta);
            }
            v2t::StreamPart::ReasoningEnd { .. } => {
                if reasoning_started {
                    eprintln!("\n</thinking>");
                    reasoning_started = false;
                }
            }
            v2t::StreamPart::TextDelta { delta, .. } => {
                if reasoning_started {
                    eprintln!("\n</thinking>");
                    reasoning_started = false;
                }
                print!("{}", delta);
            }
            v2t::StreamPart::ToolInputStart { id, tool_name, .. } => {
                if reasoning_started {
                    eprintln!("\n</thinking>");
                    reasoning_started = false;
                }
                eprintln!("\n[tool] start {} ({})", id, tool_name);
            }
            v2t::StreamPart::ToolInputDelta { id, delta, .. } => {
                if reasoning_started {
                    eprintln!("\n</thinking>");
                    reasoning_started = false;
                }
                eprintln!("\n[tool] {} += {}", id, delta);
            }
            v2t::StreamPart::ToolInputEnd { id, .. } => {
                if reasoning_started {
                    eprintln!("\n</thinking>");
                    reasoning_started = false;
                }
                eprintln!("\n[tool] end {}", id);
            }
            v2t::StreamPart::Finish { usage, .. } => {
                if reasoning_started {
                    eprintln!("\n</thinking>");
                    reasoning_started = false;
                }
                eprintln!(
                    "\n[token-usage] input:{} output:{} total:{}",
                    usage.input_tokens.unwrap_or(0),
                    usage.output_tokens.unwrap_or(0),
                    usage.total_tokens.unwrap_or(0)
                );
                break;
            }
            v2t::StreamPart::Error { error } => {
                if reasoning_started {
                    eprintln!("\n</thinking>");
                    reasoning_started = false;
                }
                eprintln!("\n[stream error] {}", error);
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
