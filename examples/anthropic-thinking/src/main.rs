use ai_sdk_rs::core::{ChatMessage, ChatRequest, Event, LanguageModel};
use ai_sdk_rs::providers::anthropic::{Anthropic, AnthropicOverridesBuilder};
use futures_util::StreamExt;
use serde_json::json;

#[tokio::main(flavor = "multi_thread")] 
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read config
    // Optional MODEL values: claude-sonnet-4-5-20250929 (default), claude-opus-4-6
    let model = std::env::var("MODEL").unwrap_or_else(|_| "claude-sonnet-4-5-20250929".to_string());
    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        eprintln!("Please set ANTHROPIC_API_KEY.");
        std::process::exit(1);
    }

    // Build request with thinking enabled (e.g., 1000 budget tokens)
    let mut req = ChatRequest::new(model.clone(), vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("Briefly explain how rainbows form."),
    ]);
    req.max_output_tokens = Some(400);
    // Option A: override builder under metadata.provider_overrides
    let overrides = AnthropicOverridesBuilder::new()
        .thinking_enabled(1000)
        .build();
    req.metadata = json!({ "provider_overrides": overrides });

    // Initialize provider and stream
    let provider: Anthropic = Anthropic::new(model).with_api_key(api_key);
    let mut stream = provider.stream(req).await?;

    let mut reasoning_started = false;
    while let Some(ev) = stream.next().await {
        match ev? {
            Event::ReasoningStart { .. } => {
                if !reasoning_started { reasoning_started = true; eprintln!("<thinking>"); }
            }
            Event::ReasoningDelta { delta } => {
                if !reasoning_started { reasoning_started = true; eprintln!("<thinking>"); }
                eprint!("{}", delta);
            }
            Event::ReasoningEnd => {
                if reasoning_started { reasoning_started = false; eprintln!("\n</thinking>"); }
            }
            Event::TextDelta { delta } => {
                // Close reasoning before visible text
                if reasoning_started { reasoning_started = false; eprintln!("\n</thinking>"); }
                print!("{}", delta);
            }
            Event::Done => {
                if reasoning_started { reasoning_started = false; eprintln!("\n</thinking>"); }
                eprintln!("\n[done]");
                break;
            }
            _ => {}
        }
    }
    Ok(())
}
