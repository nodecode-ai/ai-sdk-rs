use ai_sdk_rs::core::{ChatRequest, Message, Role, LanguageModel};
use ai_sdk_rs::providers::openai_compatible::OpenAICompatible as OpenAI;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GitHub Models provider
    // Requires GITHUB_TOKEN environment variable to be set
    let provider = OpenAI::github_models("gpt-4.1-nano");
    
    // Alternative: manually configure OpenAI provider for GitHub Models
    // let provider = OpenAI::new("gpt-4.1-nano")
    //     .with_base_url("https://models.github.com")
    //     .with_endpoint_path("/inference/chat/completions")
    //     .with_api_key(std::env::var("GITHUB_TOKEN")?)
    //     .with_extra_header("X-GitHub-Api-Version", "2022-11-28");
    
    // Create a chat request
    let request = ChatRequest {
        model: "gpt-4.1-nano".to_string(),
        messages: vec![
            Message::system("You are a helpful assistant."),
            Message::user("What is the capital of France?"),
        ],
        temperature: Some(0.7),
        max_output_tokens: Some(100),
        tools: vec![],
        metadata: None,
    };
    
    // Generate response
    let response = provider.generate(request).await?;
    println!("Response: {}", response);
    
    Ok(())
}
