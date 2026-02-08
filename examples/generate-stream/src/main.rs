use ai_sdk_rs::core::types as v2t;
use ai_sdk_rs::core::LanguageModel;
use ai_sdk_rs::providers::openai::responses::language_model::OpenAIResponsesLanguageModel;
use anyhow::Result;
use futures_util::StreamExt;

// Run with:
//   OPENAI_API_KEY=sk-... cargo run -p generate-stream
// Optional:
//   OPENAI_MODEL=gpt-4o     (default)
//   OPENAI_BASE_URL=https://api.openai.com (or proxy)

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let base_url = std::env::var("OPENAI_BASE_URL").ok();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    if api_key.is_empty() {
        eprintln!("OPENAI_API_KEY is not set. Set it in your environment.");
        std::process::exit(1);
    }

    let lm = OpenAIResponsesLanguageModel::create_simple(model, base_url, api_key);

    // Build v2 call options
    let options = v2t::CallOptions {
        prompt: vec![
            v2t::PromptMessage::System {
                content: "You are a friendly assistant!".into(),
                provider_options: None,
            },
            v2t::PromptMessage::User {
                content: vec![v2t::UserPart::Text {
                    text: "Yo".into(),
                    provider_options: None,
                }],
                provider_options: None,
            },
        ],
        ..Default::default()
    };

    let mut stream = lm.do_stream(options).await?.stream;

    // Aggregate token usage and print once at the end.
    let mut input_tokens: Option<usize> = None;
    let mut output_tokens: Option<usize> = None;
    let mut total_tokens: Option<usize> = None;
    let mut cache_read_tokens: Option<usize> = None;
    let mut cache_write_tokens: Option<usize> = None;
    let mut reasoning_started: bool = false;

    while let Some(part) = stream.next().await {
        match part? {
            v2t::StreamPart::TextDelta { delta, .. } => {
                // Close any open thinking block before user-visible text starts
                if reasoning_started {
                    eprintln!("\n</thinking>");
                    reasoning_started = false;
                }
                print!("{}", delta);
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            v2t::StreamPart::ToolInputStart { id, tool_name, .. } => {
                eprintln!("\n[tool] start {} ({})", id, tool_name);
            }
            v2t::StreamPart::ToolInputDelta { id, delta, .. } => {
                eprintln!("\n[tool] {} += {}", id, delta);
            }
            v2t::StreamPart::ToolInputEnd { id, .. } => {
                eprintln!("\n[tool] end {}", id);
            }
            v2t::StreamPart::ReasoningDelta { delta, .. } => {
                if !reasoning_started {
                    eprintln!("<thinking>");
                    reasoning_started = true;
                }
                eprint!("{}", delta);
                let _ = std::io::Write::flush(&mut std::io::stderr());
            }
            v2t::StreamPart::Raw { .. } => {
                // Raw provider chunk (advanced); ignored in example
            }
            v2t::StreamPart::ReasoningStart { .. } => {}
            v2t::StreamPart::ReasoningEnd { .. } => {}
            v2t::StreamPart::TextStart { .. } => {}
            v2t::StreamPart::TextEnd { .. } => {}
            v2t::StreamPart::ToolCall(_) => {}
            v2t::StreamPart::ToolResult { .. } => {}
            v2t::StreamPart::ToolApprovalRequest { .. } => {}
            v2t::StreamPart::ResponseMetadata { .. } => {}
            v2t::StreamPart::Error { error } => {
                eprintln!("\n[stream error] {}", error);
            }
            v2t::StreamPart::Finish { usage, .. } => {
                println!();
                if reasoning_started {
                    eprintln!("\n</thinking>");
                }
                input_tokens = usage.input_tokens.map(|v| v as usize);
                output_tokens = usage.output_tokens.map(|v| v as usize);
                total_tokens = usage.total_tokens.map(|v| v as usize);
                cache_read_tokens = usage.cached_input_tokens.map(|v| v as usize);
                break;
            }
            v2t::StreamPart::StreamStart { .. }
            | v2t::StreamPart::ReasoningSignature { .. }
            | v2t::StreamPart::SourceUrl { .. }
            | v2t::StreamPart::File { .. } => {}
        }
    }

    // Print token usage summary once, if available.
    if input_tokens.is_some() || output_tokens.is_some() || total_tokens.is_some() {
        eprintln!(
            "[token-usage] input: {} | output: {} | total: {}{}{}",
            input_tokens
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".into()),
            output_tokens
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".into()),
            total_tokens
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".into()),
            cache_read_tokens
                .map(|v| format!(" | cache_read: {}", v))
                .unwrap_or_default(),
            cache_write_tokens
                .map(|v| format!(" | cache_write: {}", v))
                .unwrap_or_default(),
        );
    }

    Ok(())
}
