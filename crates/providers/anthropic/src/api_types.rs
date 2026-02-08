use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// Mirrors packages/anthropic/src/anthropic-api-types.ts

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicCacheControl {
    Ephemeral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicUserContent {
    Text(AnthropicTextContent),
    Image(AnthropicImageContent),
    Document(AnthropicDocumentContent),
    ToolResult(AnthropicToolResultContent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicAssistantContent {
    Text(AnthropicTextContent),
    Thinking(AnthropicThinkingContent),
    RedactedThinking(AnthropicRedactedThinkingContent),
    ToolUse(AnthropicToolCallContent),
    ServerToolUse(AnthropicServerToolUseContent),
    WebSearchToolResult(AnthropicWebSearchToolResultContent),
    CodeExecutionToolResult(AnthropicCodeExecutionToolResultContent),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicTextContent {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicThinkingContent {
    pub thinking: String,
    pub signature: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicRedactedThinkingContent {
    pub data: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
    Text { media_type: String, data: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicImageContent {
    pub source: AnthropicContentSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicDocumentContent {
    pub source: AnthropicContentSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations: Option<AnthropicDocumentCitations>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicDocumentCitations {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicToolCallContent {
    pub id: String,
    pub name: String,
    pub input: JsonValue,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicServerToolUseContent {
    pub id: String,
    pub name: String, // 'web_search' | 'code_execution'
    pub input: JsonValue,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicToolResultContent {
    pub tool_use_id: String,
    pub content: JsonValue, // string | Array<Text|Image>
    #[serde(default)]
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicWebSearchResultItem {
    pub url: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_age: Option<String>,
    pub encrypted_content: String,
    pub r#type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicWebSearchToolResultContent {
    pub tool_use_id: String,
    pub content: JsonValue, // array of AnthropicWebSearchResultItem or error object
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct AnthropicCodeExecutionToolResultContent {
    pub tool_use_id: String,
    pub content: JsonValue, // code_execution_result or error object
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}

// Tool definitions

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicTool {
    #[serde(rename = "function")]
    Function {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        input_schema: JsonValue,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    #[serde(rename = "computer_20250124")]
    Computer20250124 {
        name: String,
        display_width_px: u32,
        display_height_px: u32,
        display_number: u32,
    },
    #[serde(rename = "computer_20241022")]
    Computer20241022 {
        name: String,
        display_width_px: u32,
        display_height_px: u32,
        display_number: u32,
    },
    #[serde(rename = "text_editor_20250124")]
    TextEditor20250124 { name: String },
    #[serde(rename = "text_editor_20241022")]
    TextEditor20241022 { name: String },
    #[serde(rename = "text_editor_20250429")]
    TextEditor20250429 { name: String },
    #[serde(rename = "text_editor_20250728")]
    TextEditor20250728 {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_characters: Option<u32>,
    },
    #[serde(rename = "bash_20250124")]
    Bash20250124 { name: String },
    #[serde(rename = "bash_20241022")]
    Bash20241022 { name: String },
    #[serde(rename = "web_search_20250305")]
    WebSearch20250305 {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_uses: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        allowed_domains: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        blocked_domains: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user_location: Option<WebSearchUserLocation>,
    },
    #[serde(rename = "web_fetch_20250910")]
    WebFetch20250910 {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_uses: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        allowed_domains: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        blocked_domains: Option<Vec<String>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<WebFetchCitations>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_content_tokens: Option<u32>,
    },
    #[serde(rename = "memory_20250818")]
    Memory20250818 { name: String },
    #[serde(rename = "tool_search_tool_regex_20251119")]
    ToolSearchToolRegex20251119 { name: String },
    #[serde(rename = "tool_search_tool_bm25_20251119")]
    ToolSearchToolBm25_20251119 { name: String },
    #[serde(rename = "code_execution_20250522")]
    CodeExecution20250522 { name: String },
    #[serde(rename = "code_execution_20250825")]
    CodeExecution20250825 { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WebSearchUserLocation {
    pub r#type: String, // 'approximate'
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct WebFetchCitations {
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicToolChoice {
    Auto {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Tool {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
}
