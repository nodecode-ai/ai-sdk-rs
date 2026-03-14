//! Unified type definitions for AI SDK chat protocols
//!
//! This crate provides the core types used across both ai-sdk-rs and clixode
//! for representing chat messages, requests, and streaming events.

pub mod embedding;
pub mod image;
pub mod json;
pub mod usage;
pub mod v2;

use serde::{Deserialize, Serialize};

/// Basic roles for chat messages.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Represents the streaming state of tool arguments.
///
/// During streaming, tool arguments may be in various states from pending
/// to complete. This enum provides explicit representation of these states
/// to ensure consistency between streaming and resumed sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum StreamingState {
    /// Initial state, no arguments received yet
    Pending,
    /// Currently receiving and accumulating arguments
    Accumulating { bytes: usize },
    /// Attempting to parse accumulated arguments
    Parsing { bytes: usize },
    /// Parsing failed with error
    Failed { error: String },
}

/// Represents tool arguments that may be streaming or complete.
///
/// This enum allows proper representation of tool arguments during
/// streaming without relying on special JSON markers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolArguments {
    /// Arguments are still being streamed
    Streaming {
        state: StreamingState,
        #[serde(skip_serializing_if = "Option::is_none")]
        raw_json: Option<String>,
    },
    /// Arguments are complete and parsed
    Complete(serde_json::Value),
}

impl ToolArguments {
    /// Returns the complete arguments if available.
    pub fn as_complete(&self) -> Option<&serde_json::Value> {
        match self {
            ToolArguments::Complete(args) => Some(args),
            _ => None,
        }
    }

    /// Returns true if arguments are complete.
    pub fn is_complete(&self) -> bool {
        matches!(self, ToolArguments::Complete(_))
    }

    /// Returns true if arguments are still streaming.
    pub fn is_streaming(&self) -> bool {
        matches!(self, ToolArguments::Streaming { .. })
    }

    /// Convert to a serde_json::Value, using empty object if streaming
    pub fn to_value(&self) -> serde_json::Value {
        match self {
            ToolArguments::Complete(v) => v.clone(),
            ToolArguments::Streaming { raw_json, .. } => raw_json
                .as_deref()
                .and_then(|s| crate::types::json::parse_json_loose(s))
                .unwrap_or_else(|| serde_json::json!({})),
        }
    }
}

/// Content parts that can appear in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text content
    Text { text: String },
    /// Persisted model reasoning/thinking content for assistant messages
    /// Providers may convert this into provider-specific request blocks (e.g., Anthropic "thinking")
    Reasoning {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        redacted_data: Option<String>,
    },
    /// Assistant requests a tool call
    ToolUse {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        name: String,
        input: serde_json::Value,
    },
    /// Assistant requests a tool call with streaming support
    /// This variant is used internally during streaming
    ToolUseStreaming {
        #[serde(rename = "id", skip_serializing_if = "Option::is_none")]
        tool_use_id: Option<String>,
        #[serde(rename = "name")]
        tool_name: String,
        #[serde(rename = "input")]
        arguments: ToolArguments,
    },
    /// User provides tool output
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
    /// Image content
    Image { data: Vec<u8>, mime_type: String },
    /// Error message
    Error { error: String },
}

impl ContentPart {
    /// Convert a streaming tool use to a regular tool use if arguments are complete
    pub fn normalize(self) -> Self {
        match self {
            ContentPart::ToolUseStreaming {
                tool_use_id,
                tool_name,
                arguments,
                ..
            } => match arguments {
                ToolArguments::Complete(input) => ContentPart::ToolUse {
                    id: tool_use_id,
                    name: tool_name,
                    input,
                },
                ToolArguments::Streaming { raw_json, .. } => ContentPart::ToolUse {
                    id: tool_use_id,
                    name: tool_name,
                    input: raw_json
                        .as_deref()
                        .and_then(|s| crate::types::json::parse_json_loose(s))
                        .unwrap_or_else(|| serde_json::json!({})),
                },
            },
            other => other,
        }
    }
}

/// A chat message with role and content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub parts: Vec<ContentPart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

// Helpers for convenience
impl ChatMessage {
    pub fn system<S: Into<String>>(s: S) -> Self {
        Self {
            role: Role::System,
            parts: vec![ContentPart::Text { text: s.into() }],
            name: None,
        }
    }

    pub fn user<S: Into<String>>(s: S) -> Self {
        Self {
            role: Role::User,
            parts: vec![ContentPart::Text { text: s.into() }],
            name: None,
        }
    }

    pub fn assistant<S: Into<String>>(s: S) -> Self {
        Self {
            role: Role::Assistant,
            parts: vec![ContentPart::Text { text: s.into() }],
            name: None,
        }
    }

    /// Concatenate all Text parts into a single String.
    pub fn text(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

/// JSON Schema specification for a tool.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolSpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub json_schema: serde_json::Value,
}

/// Normalized chat request that providers accept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub tools: Vec<ToolSpec>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            max_output_tokens: None,
            metadata: serde_json::Value::Null,
            tools: vec![],
        }
    }
}

/// Streamed events for UI or SSE serialization.
///
/// See this enum plus `v2::StreamPart` for normalized streaming semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    TextDelta {
        delta: String,
    },
    /// Non-visible model reasoning/thinking stream.
    /// Concatenate deltas for a separate reasoning view.
    ReasoningDelta {
        delta: String,
    },
    /// Reasoning lifecycle start. Implementations typically emit a single stream id.
    ReasoningStart {
        id: String,
    },
    /// Reasoning lifecycle end for the given stream id.
    ReasoningEnd,
    /// Standardized token usage event.
    ///
    /// Providers SHOULD emit this when token usage information is available.
    /// Providers MAY also emit a Data { key: "usage" } event containing the raw provider payload.
    Usage {
        usage: TokenUsage,
    },
    /// Tool call lifecycle: start of a tool call
    ToolCallStart {
        id: String,
        name: String,
    },
    /// Tool call lifecycle: streaming arguments delta
    ToolCallDelta {
        id: String,
        args_json: String,
    },
    /// Tool call lifecycle: end of a tool call
    ToolCallEnd {
        id: String,
    },
    Data {
        key: String,
        value: serde_json::Value,
    },
    Error {
        message: String,
    },
    /// Indicates a retry attempt is being made
    Retrying {
        provider: String,
        attempt: u32,
        max_attempts: u32,
        error: String,
        delay_secs: u64,
    },
    /// Provider raw chunk passthrough (advanced; default off).
    /// When enabled, providers may emit raw provider-originated chunks for
    /// advanced integrations. The shape depends on the provider; when the
    /// source is JSON, it is forwarded as a JSON value, otherwise as a string.
    Raw {
        #[serde(rename = "raw_value")]
        raw_value: serde_json::Value,
    },
    Done,
}

/// Token usage information for a message.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<usize>,
}

impl TokenUsage {
    /// Creates a new TokenUsage instance with the given input and output tokens.
    pub fn new(input_tokens: usize, output_tokens: usize) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
        }
    }
}

// Catalog types module
pub mod catalog;

// Re-export commonly used types at the crate root
pub use self::{ContentPart as MessageContent, Role as MessageRole};
