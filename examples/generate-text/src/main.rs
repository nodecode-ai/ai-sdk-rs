use ai_sdk_rs::core::types as v2t;
use ai_sdk_rs::core::LanguageModel;
use ai_sdk_rs::providers::openai::OpenAIResponsesLanguageModel;
use anyhow::Result;

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
                    text: "Why is the sky blue?".into(),
                    provider_options: None,
                }],
                provider_options: None,
            },
        ],
        ..Default::default()
    }
}

fn collect_text(content: &[v2t::Content]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            v2t::Content::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = build_model()?;
    let response = model.do_generate(build_options()).await?;
    let text = collect_text(&response.content);

    if text.is_empty() {
        anyhow::bail!("model returned no text content");
    }

    println!("{}", text);
    Ok(())
}
