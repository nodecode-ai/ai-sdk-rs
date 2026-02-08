use ai_sdk_rs::core::{ChatMessage, ChatRequest, Event, LanguageModel};
use ai_sdk_rs::providers::anthropic::{Anthropic, AnthropicOverridesBuilder};
use futures_util::StreamExt;
use serde_json::json;

// Run with:
//   ANTHROPIC_OAUTH_BEARER="Bearer sk-ant-..." cargo run -p anthropic-hello-twice
// Optional: MODEL=claude-sonnet-4-5-20250929 or MODEL=claude-opus-4-6

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default to the requested model; override with MODEL env if needed.
    // Using: "claude-sonnet-4-5-20250929" (thinking-capable)
    let model = std::env::var("MODEL").unwrap_or_else(|_| "claude-sonnet-4-5-20250929".to_string());
    // Use OAuth bearer for Authorization, which also enables anthropic-beta header.
    // Set env var ANTHROPIC_OAUTH_BEARER to the full value, e.g.:
    //   export ANTHROPIC_OAUTH_BEARER="Bearer sk-ant-..."
    let oauth_bearer = std::env::var("ANTHROPIC_OAUTH_BEARER").unwrap_or_default();
    if oauth_bearer.is_empty() {
        eprintln!("Please set ANTHROPIC_OAUTH_BEARER to your 'Bearer ...' token.");
        std::process::exit(1);
    }

    let provider: Anthropic = Anthropic::new(model).with_oauth_bearer(oauth_bearer);

    // Turn 1: say "yo", with thinking enabled (small budget).
    let mut req = ChatRequest::new(
        provider.id(),
        vec![
            ChatMessage::system("You are a terse assistant. Reply with exactly what the user writes, no extra text."),
            ChatMessage::user("yo"),
        ],
    );
    req.max_output_tokens = Some(20);
    // Anthropic requires a minimum thinking budget of 1024 tokens.
    let overrides = AnthropicOverridesBuilder::new().thinking_enabled(1024).build();
    req.metadata = json!({ "provider_overrides": overrides });

    // Stream response to capture reasoning deltas and signature.
    let mut stream = provider.stream(req).await?;

    let mut visible = String::new();
    let mut reasoning = String::new();
    let mut signature: Option<String> = None;
    let mut reasoning_started = false;

    while let Some(ev) = stream.next().await {
        match ev? {
            Event::TextDelta { delta } => {
                // Close thinking block before printing visible text
                if reasoning_started {
                    eprintln!("\n</thinking>");
                    if let Some(sig) = &signature { eprintln!("[signature] {}", sig); }
                    reasoning_started = false;
                }
                print!("{}", delta);
                let _ = std::io::Write::flush(&mut std::io::stdout());
                visible.push_str(&delta);
            }
            Event::ReasoningDelta { delta } => {
                if !reasoning_started {
                    eprintln!("<thinking>");
                    reasoning_started = true;
                }
                eprint!("{}", delta);
                let _ = std::io::Write::flush(&mut std::io::stderr());
                reasoning.push_str(&delta);
            }
            Event::Data { key, value } if key == "reasoning_signature" => {
                if let Some(sig) = value.get("signature").and_then(|v| v.as_str()) {
                    signature = Some(sig.to_string());
                }
            }
            Event::Done => {
                if reasoning_started {
                    eprintln!("\n</thinking>");
                    if let Some(sig) = &signature { eprintln!("[signature] {}", sig); }
                    reasoning_started = false;
                }
                break;
            }
            _ => {}
        }
    }

    // Simple assertions: reasoning text and signature must be present and non-empty.
    if reasoning.trim().is_empty() {
        eprintln!("\n[assertion failed] Missing or empty reasoning text.");
        std::process::exit(2);
    }
    if signature.as_deref().unwrap_or("").trim().is_empty() {
        eprintln!("\n[assertion failed] Missing reasoning signature.");
        std::process::exit(3);
    }

    // Optional: ensure visible reply is exactly "yo" (tolerate trailing newline differences).
    let norm = visible.trim();
    if norm != "yo" {
        eprintln!("\n[warning] First turn not exactly 'yo': {:?}", norm);
    }

    // --- Second turn: build on the first ---
    // Use the first assistant reply as context and ask to repeat it exactly.
    // This demonstrates multi-turn state carried via prior messages.
    eprintln!("\n----- second turn (say 'yo2') -----");
    let mut req2 = ChatRequest::new(
        provider.id(),
        vec![
            // Keep the same system rule for strict output shape.
            ChatMessage::system("You are a terse assistant. Reply with exactly what the user writes, no extra text."),
            ChatMessage::user("yo"),
            ChatMessage::assistant(visible.clone()),
            ChatMessage::user("yo2"),
        ],
    );
    req2.max_output_tokens = Some(20);
    req2.metadata = json!({ "provider_overrides": overrides });

    let mut stream2 = provider.stream(req2).await?;

    let mut visible2 = String::new();
    let mut reasoning2 = String::new();
    let mut signature2: Option<String> = None;
    let mut reasoning2_started = false;

    while let Some(ev) = stream2.next().await {
        match ev? {
            Event::TextDelta { delta } => {
                if reasoning2_started {
                    eprintln!("\n</thinking>");
                    if let Some(sig) = &signature2 { eprintln!("[signature] {}", sig); }
                    reasoning2_started = false;
                }
                print!("{}", delta);
                let _ = std::io::Write::flush(&mut std::io::stdout());
                visible2.push_str(&delta);
            }
            Event::ReasoningDelta { delta } => {
                if !reasoning2_started {
                    eprintln!("<thinking>");
                    reasoning2_started = true;
                }
                eprint!("{}", delta);
                let _ = std::io::Write::flush(&mut std::io::stderr());
                reasoning2.push_str(&delta);
            }
            Event::Data { key, value } if key == "reasoning_signature" => {
                if let Some(sig) = value.get("signature").and_then(|v| v.as_str()) {
                    signature2 = Some(sig.to_string());
                }
            }
            Event::Done => {
                if reasoning2_started {
                    eprintln!("\n</thinking>");
                    if let Some(sig) = &signature2 { eprintln!("[signature] {}", sig); }
                    reasoning2_started = false;
                }
                break;
            }
            _ => {}
        }
    }

    // Validate second turn also produced reasoning and signature.
    if reasoning2.trim().is_empty() {
        eprintln!("\n[assertion failed] Second turn missing or empty reasoning text.");
        std::process::exit(4);
    }
    if signature2.as_deref().unwrap_or("").trim().is_empty() {
        eprintln!("\n[assertion failed] Second turn missing reasoning signature.");
        std::process::exit(5);
    }

    // Optional: second reply should be exactly "yo2".
    let norm2 = visible2.trim();
    if norm2 != "yo2" {
        eprintln!("\n[warning] Second turn not exactly 'yo2': {:?}", norm2);
    }

    Ok(())
}
