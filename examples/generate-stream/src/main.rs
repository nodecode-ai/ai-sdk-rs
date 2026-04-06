use std::io::{self, Write};

use ai_sdk_rs::core::types as v2t;
use ai_sdk_rs::core::LanguageModel;
use ai_sdk_rs::core::SdkError;
use ai_sdk_rs::providers::openai::OpenAIResponsesLanguageModel;
use anyhow::Result;
use futures_util::{Stream, StreamExt};

// Run with:
//   OPENAI_API_KEY=sk-... cargo run -p generate-stream
// Optional:
//   OPENAI_MODEL=gpt-4o     (default)
//   OPENAI_BASE_URL=https://api.openai.com (or proxy)

#[derive(Default)]
struct UsageSummary {
    input_tokens: Option<usize>,
    output_tokens: Option<usize>,
    total_tokens: Option<usize>,
    cache_read_tokens: Option<usize>,
}

impl UsageSummary {
    fn has_visible_tokens(&self) -> bool {
        self.input_tokens.is_some() || self.output_tokens.is_some() || self.total_tokens.is_some()
    }
}

impl From<v2t::Usage> for UsageSummary {
    fn from(usage: v2t::Usage) -> Self {
        Self {
            input_tokens: usage.input_tokens.map(|value| value as usize),
            output_tokens: usage.output_tokens.map(|value| value as usize),
            total_tokens: usage.total_tokens.map(|value| value as usize),
            cache_read_tokens: usage.cached_input_tokens.map(|value| value as usize),
        }
    }
}

#[derive(Default)]
struct StreamDisplayState {
    reasoning_started: bool,
    usage: UsageSummary,
}

impl StreamDisplayState {
    fn close_reasoning_block(&mut self) {
        if self.reasoning_started {
            eprintln!("\n</thinking>");
            self.reasoning_started = false;
        }
    }

    fn print_text_delta(&mut self, delta: &str) {
        self.close_reasoning_block();
        print!("{delta}");
        let _ = io::stdout().flush();
    }

    fn print_reasoning_delta(&mut self, delta: &str) {
        if !self.reasoning_started {
            eprintln!("<thinking>");
            self.reasoning_started = true;
        }
        eprint!("{delta}");
        let _ = io::stderr().flush();
    }

    fn record_finish(&mut self, usage: v2t::Usage) {
        println!();
        self.close_reasoning_block();
        self.usage = usage.into();
    }
}

fn build_model() -> Result<OpenAIResponsesLanguageModel> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let base_url = std::env::var("OPENAI_BASE_URL").ok();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    if api_key.is_empty() {
        eprintln!("OPENAI_API_KEY is not set. Set it in your environment.");
        std::process::exit(1);
    }

    let mut builder = OpenAIResponsesLanguageModel::builder(model);
    if let Some(base_url) = base_url {
        builder = builder.with_base_url(base_url);
    }
    Ok(builder.with_api_key(api_key).build()?)
}

fn build_options() -> v2t::CallOptions {
    v2t::CallOptions {
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
    }
}

fn handle_stream_part(part: v2t::StreamPart, state: &mut StreamDisplayState) -> bool {
    match part {
        v2t::StreamPart::TextDelta { delta, .. } => state.print_text_delta(&delta),
        v2t::StreamPart::ToolInputStart { id, tool_name, .. } => {
            eprintln!("\n[tool] start {id} ({tool_name})");
        }
        v2t::StreamPart::ToolInputDelta { id, delta, .. } => {
            eprintln!("\n[tool] {id} += {delta}");
        }
        v2t::StreamPart::ToolInputEnd { id, .. } => eprintln!("\n[tool] end {id}"),
        v2t::StreamPart::ReasoningDelta { delta, .. } => state.print_reasoning_delta(&delta),
        v2t::StreamPart::Error { error } => eprintln!("\n[stream error] {error}"),
        v2t::StreamPart::Finish { usage, .. } => {
            state.record_finish(usage);
            return true;
        }
        v2t::StreamPart::Raw { .. }
        | v2t::StreamPart::ReasoningStart { .. }
        | v2t::StreamPart::ReasoningEnd { .. }
        | v2t::StreamPart::TextStart { .. }
        | v2t::StreamPart::TextEnd { .. }
        | v2t::StreamPart::ToolCall(_)
        | v2t::StreamPart::ToolResult { .. }
        | v2t::StreamPart::ToolApprovalRequest { .. }
        | v2t::StreamPart::ResponseMetadata { .. }
        | v2t::StreamPart::StreamStart { .. }
        | v2t::StreamPart::ReasoningSignature { .. }
        | v2t::StreamPart::SourceUrl { .. }
        | v2t::StreamPart::File { .. } => {}
    }
    false
}

async fn consume_stream<S>(mut stream: S) -> Result<UsageSummary>
where
    S: Stream<Item = std::result::Result<v2t::StreamPart, SdkError>> + Unpin,
{
    let mut state = StreamDisplayState::default();

    while let Some(part) = stream.next().await {
        if handle_stream_part(part?, &mut state) {
            break;
        }
    }

    Ok(state.usage)
}

fn print_usage_summary(summary: &UsageSummary) {
    if !summary.has_visible_tokens() {
        return;
    }

    eprintln!(
        "[token-usage] input: {} | output: {} | total: {}{}",
        summary
            .input_tokens
            .map(|value| value.to_string())
            .unwrap_or_else(|| "?".into()),
        summary
            .output_tokens
            .map(|value| value.to_string())
            .unwrap_or_else(|| "?".into()),
        summary
            .total_tokens
            .map(|value| value.to_string())
            .unwrap_or_else(|| "?".into()),
        summary
            .cache_read_tokens
            .map(|value| format!(" | cache_read: {value}"))
            .unwrap_or_default(),
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = build_model()?;
    let options = build_options();
    let usage = consume_stream(model.do_stream(options).await?.stream).await?;
    print_usage_summary(&usage);
    Ok(())
}
