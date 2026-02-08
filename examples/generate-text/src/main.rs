use anyhow::Result;
use ai_sdk_rs::core::{ChatMessage, ChatRequest, LanguageModel};
use ai_sdk_rs::providers::openai_compatible::OpenAICompatible as OpenAI;

// Run with:
//   OPENAI_API_KEY=sk-... cargo run -p generate-text
// Optional:
//   OPENAI_MODEL=gpt-4o     (default)
//   OPENAI_BASE_URL=https://api.openai.com (or proxy)
//
// Mirrors:
// import { generateText } from 'ai';
// import { openai } from '@ai-sdk/openai';
// const { text } = await generateText({
//   model: openai('gpt-4o'),
//   system: 'You are a friendly assistant!',
//   prompt: 'Why is the sky blue?',
// });
// console.log(text);

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let base_url = std::env::var("OPENAI_BASE_URL").ok();
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    if api_key.is_empty() {
        eprintln!("OPENAI_API_KEY is not set. Set it in your environment.");
        std::process::exit(1);
    }

    let mut provider = OpenAI::new(model);
    if let Some(b) = base_url { provider = provider.with_base_url(b); }
    provider = provider.with_api_key(api_key);

    let req = ChatRequest::new(
        provider.id(),
        vec![
            ChatMessage::system("You are a friendly assistant!"),
            ChatMessage::user("Why is the sky blue?"),
        ],
    );

    let text = provider.generate(req).await?;
    println!("{}", text);
    Ok(())
}
