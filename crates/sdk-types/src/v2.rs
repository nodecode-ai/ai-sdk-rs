//! LanguageModel V2 parity types inspired by Vercel AI SDK.
//! These types are provider-agnostic and designed for interop with adapters.

use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ---------- Provider plumbing ----------

/// Provider-specific input options passed through to providers.
/// Outer key is the provider id; inner keys are provider-defined option names.
use std::collections::HashMap;

pub type ProviderOptions = HashMap<String, HashMap<String, JsonValue>>;

/// Provider-specific output metadata passed through from providers.
/// Outer key is the provider id; inner keys are provider-defined metadata keys.
pub type ProviderMetadata =
    std::collections::HashMap<String, std::collections::HashMap<String, JsonValue>>;

/// HTTP headers map for response metadata.
pub type Headers = HashMap<String, String>;

pub(crate) fn headers_is_empty(map: &HashMap<String, String>) -> bool {
    map.is_empty()
}

pub(crate) fn provider_options_is_empty(map: &ProviderOptions) -> bool {
    map.is_empty()
}

pub(crate) fn bool_is_false(value: &bool) -> bool {
    !*value
}

// ---------- Prompt ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum PromptMessage {
    System {
        content: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerOptions"
        )]
        provider_options: Option<ProviderOptions>,
    },
    User {
        content: Vec<UserPart>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerOptions"
        )]
        provider_options: Option<ProviderOptions>,
    },
    Assistant {
        content: Vec<AssistantPart>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerOptions"
        )]
        provider_options: Option<ProviderOptions>,
    },
    Tool {
        content: Vec<ToolMessagePart>,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerOptions"
        )]
        provider_options: Option<ProviderOptions>,
    },
}

pub type Prompt = Vec<PromptMessage>;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum UserPart {
    Text {
        text: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerOptions"
        )]
        provider_options: Option<ProviderOptions>,
    },
    File {
        filename: Option<String>,
        data: DataContent,
        #[serde(rename = "mediaType")]
        media_type: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerOptions"
        )]
        provider_options: Option<ProviderOptions>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum AssistantPart {
    Text {
        text: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerOptions"
        )]
        provider_options: Option<ProviderOptions>,
    },
    Reasoning {
        text: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerOptions"
        )]
        provider_options: Option<ProviderOptions>,
    },
    File {
        filename: Option<String>,
        data: DataContent,
        #[serde(rename = "mediaType")]
        media_type: String,
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            rename = "providerOptions"
        )]
        provider_options: Option<ProviderOptions>,
    },
    ToolCall(ToolCallPart),
    ToolResult(ToolResultPart),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum DataContent {
    /// Base64-encoded data string
    Base64 { base64: String },
    /// Raw bytes
    Bytes {
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
    },
    /// URL string
    Url { url: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallPart {
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    /// Stringified JSON input
    #[serde(rename = "input")]
    pub input: String,
    #[serde(default, rename = "providerExecuted")]
    pub provider_executed: bool,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerMetadata"
    )]
    pub provider_metadata: Option<ProviderMetadata>,
    #[serde(default, skip_serializing_if = "bool_is_false")]
    pub dynamic: bool,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerOptions"
    )]
    pub provider_options: Option<ProviderOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolResultOutput {
    Text { value: String },
    Json { value: JsonValue },
    ErrorText { value: String },
    ErrorJson { value: JsonValue },
    Content { value: Vec<ToolResultInlineContent> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum ToolResultInlineContent {
    Text {
        text: String,
    },
    Media {
        data: String,
        #[serde(rename = "mediaType")]
        media_type: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultPart {
    #[serde(rename = "type", default = "tool_result_part_type_default")]
    pub r#type: ToolResultPartType,
    #[serde(rename = "toolCallId")]
    pub tool_call_id: String,
    #[serde(rename = "toolName")]
    pub tool_name: String,
    #[serde(rename = "output")]
    pub output: ToolResultOutput,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerOptions"
    )]
    pub provider_options: Option<ProviderOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolResultPartType {
    ToolResult,
}

fn tool_result_part_type_default() -> ToolResultPartType {
    ToolResultPartType::ToolResult
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolMessagePart {
    ToolResult(ToolResultPart),
    ToolApprovalResponse(ToolApprovalResponsePart),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolApprovalResponsePart {
    #[serde(rename = "type", default = "tool_approval_response_type_default")]
    pub r#type: ToolApprovalResponsePartType,
    #[serde(rename = "approvalId")]
    pub approval_id: String,
    pub approved: bool,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerOptions"
    )]
    pub provider_options: Option<ProviderOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolApprovalResponsePartType {
    ToolApprovalResponse,
}

fn tool_approval_response_type_default() -> ToolApprovalResponsePartType {
    ToolApprovalResponsePartType::ToolApprovalResponse
}

// ---------- Call options ----------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CallOptions {
    pub prompt: Prompt,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default)]
    pub include_raw_chunks: bool,
    #[serde(default, skip_serializing_if = "headers_is_empty")]
    pub headers: HashMap<String, String>,
    #[serde(default, skip_serializing_if = "provider_options_is_empty")]
    pub provider_options: ProviderOptions,
}

impl CallOptions {
    pub fn new(prompt: Prompt) -> Self {
        Self {
            prompt,
            ..Default::default()
        }
    }
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }
    pub fn with_max_output_tokens(mut self, n: u32) -> Self {
        self.max_output_tokens = Some(n);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ResponseFormat {
    Text,
    Json {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        schema: Option<JsonValue>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionTool {
    #[serde(rename = "type", default = "function_tool_type_default")]
    pub r#type: FunctionToolType,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: JsonValue,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "providerOptions"
    )]
    pub provider_options: Option<ProviderOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FunctionToolType {
    Function,
}

fn function_tool_type_default() -> FunctionToolType {
    FunctionToolType::Function
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Tool {
    Function(FunctionTool),
    Provider(ProviderTool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderTool {
    #[serde(rename = "type", default = "provider_tool_type_default")]
    pub r#type: ProviderToolType,
    /// The provider tool id, formatted as "<provider>.<tool>".
    pub id: String,
    /// The custom tool name exposed to the model.
    pub name: String,
    /// Provider-defined tool arguments.
    pub args: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderToolType {
    Provider,
}

fn provider_tool_type_default() -> ProviderToolType {
    ProviderToolType::Provider
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Tool { name: String },
}

// ---------- Warnings / finish / usage ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum CallWarning {
    UnsupportedSetting {
        setting: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    UnsupportedTool {
        tool_name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    Other {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Error,
    Other,
    Unknown,
}

impl Default for FinishReason {
    fn default() -> Self {
        FinishReason::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
}

// ---------- Model outputs ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Content {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    Reasoning {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    File {
        media_type: String,
        data: String,
    },
    SourceUrl {
        id: String,
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolCall(ToolCallPart),
    ToolApprovalRequest {
        #[serde(rename = "approvalId")]
        approval_id: String,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        result: JsonValue,
        #[serde(default)]
        is_error: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
}

// ---------- Streaming ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum StreamPart {
    // Text
    TextStart {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    TextDelta {
        id: String,
        delta: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    TextEnd {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    // Reasoning
    ReasoningStart {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ReasoningDelta {
        id: String,
        delta: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ReasoningEnd {
        id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ReasoningSignature {
        signature: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    // Tool input + calls/results
    ToolInputStart {
        id: String,
        tool_name: String,
        #[serde(default, rename = "providerExecuted")]
        provider_executed: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolInputDelta {
        id: String,
        delta: String,
        #[serde(default, rename = "providerExecuted")]
        provider_executed: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolInputEnd {
        id: String,
        #[serde(default, rename = "providerExecuted")]
        provider_executed: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolCall(ToolCallPart),
    ToolApprovalRequest {
        #[serde(rename = "approvalId")]
        approval_id: String,
        #[serde(rename = "toolCallId")]
        tool_call_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        result: JsonValue,
        #[serde(default)]
        is_error: bool,
        #[serde(default, skip_serializing_if = "bool_is_false")]
        preliminary: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    // Files and sources
    File {
        media_type: String,
        data: String,
    },
    SourceUrl {
        id: String,
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    // Stream lifecycle and metadata
    StreamStart {
        warnings: Vec<CallWarning>,
    },
    ResponseMetadata {
        #[serde(flatten)]
        meta: ResponseMetadata,
    },
    Finish {
        usage: Usage,
        finish_reason: FinishReason,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_metadata: Option<ProviderMetadata>,
    },
    // Raw and error passthroughs
    Raw {
        raw_value: JsonValue,
    },
    Error {
        error: JsonValue,
    },
}
