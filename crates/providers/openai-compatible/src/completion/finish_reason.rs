use crate::ai_sdk_types::v2::FinishReason;

pub fn map_openai_compatible_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason.unwrap_or("") {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::Length,
        "content_filter" => FinishReason::ContentFilter,
        "function_call" | "tool_calls" => FinishReason::ToolCalls,
        _ => FinishReason::Unknown,
    }
}
