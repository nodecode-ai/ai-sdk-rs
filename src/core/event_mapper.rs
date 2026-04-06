use std::collections::{HashMap, HashSet};

use crate::ai_sdk_types::v2 as v2t;
use crate::ai_sdk_types::Event as ProviderEvent;
use async_stream::try_stream;
use futures_core::Stream;
use futures_util::StreamExt;

use crate::ai_sdk_core::error::SdkError;
use crate::ai_sdk_core::v2::PartStream;

pub type ProviderMetadata = HashMap<String, HashMap<String, serde_json::Value>>;

pub struct StreamNormalizationState<Extra> {
    pub text_open: Option<String>,
    pub reasoning_open: Option<String>,
    pub tool_args: HashMap<String, String>,
    pub tool_names: HashMap<String, String>,
    pub usage: v2t::Usage,
    pub has_tool_calls: bool,
    pub extra: Extra,
}

pub type EventMapperState<Extra> = StreamNormalizationState<Extra>;

type MetaFn<Extra> =
    Box<dyn FnMut(&mut StreamNormalizationState<Extra>) -> Option<ProviderMetadata> + Send>;
type ToolMetaFn<Extra> = Box<
    dyn FnMut(&mut StreamNormalizationState<Extra>, &str, &str) -> Option<ProviderMetadata> + Send,
>;
type ToolEndMetaFn<Extra> =
    Box<dyn FnMut(&mut StreamNormalizationState<Extra>, &str) -> Option<ProviderMetadata> + Send>;
type DataFn<Extra> = Box<
    dyn FnMut(
            &mut StreamNormalizationState<Extra>,
            &str,
            &serde_json::Value,
        ) -> Option<Vec<v2t::StreamPart>>
        + Send,
>;
type FinishFn<Extra> = Box<
    dyn Fn(&StreamNormalizationState<Extra>) -> (v2t::FinishReason, Option<ProviderMetadata>)
        + Send,
>;

/// Hooks for provider-specific stream handling.
#[derive(Default)]
pub struct EventMapperHooks<Extra> {
    pub text_start_metadata: Option<MetaFn<Extra>>,
    pub reasoning_start_metadata: Option<MetaFn<Extra>>,
    pub tool_start_metadata: Option<ToolMetaFn<Extra>>,
    pub tool_end_metadata: Option<ToolEndMetaFn<Extra>>,
    pub data: Option<DataFn<Extra>>,
    pub finish: Option<FinishFn<Extra>>,
}

pub struct EventMapperConfig<Extra> {
    pub warnings: Vec<v2t::CallWarning>,
    pub treat_tool_names_as_text: HashSet<String>,
    pub default_text_id: &'static str,
    pub finish_reason_fallback: v2t::FinishReason,
    pub initial_extra: Extra,
    pub hooks: EventMapperHooks<Extra>,
}

impl<Extra> StreamNormalizationState<Extra> {
    pub fn new(extra: Extra) -> Self {
        Self {
            text_open: None,
            reasoning_open: None,
            tool_args: HashMap::new(),
            tool_names: HashMap::new(),
            usage: v2t::Usage::default(),
            has_tool_calls: false,
            extra,
        }
    }

    pub fn open_text(
        &mut self,
        id: String,
        provider_metadata: Option<ProviderMetadata>,
    ) -> Vec<v2t::StreamPart> {
        let mut parts = Vec::new();
        if self.text_open.as_ref() != Some(&id) {
            if let Some(open_id) = self.text_open.replace(id.clone()) {
                parts.push(v2t::StreamPart::TextEnd {
                    id: open_id,
                    provider_metadata: None,
                });
            }
            parts.push(v2t::StreamPart::TextStart {
                id,
                provider_metadata,
            });
        }
        parts
    }

    pub fn push_text_delta(
        &mut self,
        id: Option<String>,
        default_text_id: &str,
        delta: String,
        start_metadata: Option<ProviderMetadata>,
        provider_metadata: Option<ProviderMetadata>,
    ) -> Vec<v2t::StreamPart> {
        let id = id
            .or_else(|| self.text_open.clone())
            .unwrap_or_else(|| default_text_id.to_string());
        let mut parts = Vec::new();
        parts.extend(self.open_text(id.clone(), start_metadata));
        parts.push(v2t::StreamPart::TextDelta {
            id,
            delta,
            provider_metadata,
        });
        parts
    }

    pub fn close_text(
        &mut self,
        provider_metadata: Option<ProviderMetadata>,
    ) -> Option<v2t::StreamPart> {
        self.text_open
            .take()
            .map(|id| self.text_end_part(id, provider_metadata))
    }

    pub fn text_end_part(
        &self,
        id: String,
        provider_metadata: Option<ProviderMetadata>,
    ) -> v2t::StreamPart {
        v2t::StreamPart::TextEnd {
            id,
            provider_metadata,
        }
    }

    pub fn open_reasoning(
        &mut self,
        id: String,
        provider_metadata: Option<ProviderMetadata>,
    ) -> Vec<v2t::StreamPart> {
        let mut parts = Vec::new();
        if self.reasoning_open.as_ref() != Some(&id) {
            if let Some(open_id) = self.reasoning_open.replace(id.clone()) {
                parts.push(v2t::StreamPart::ReasoningEnd {
                    id: open_id,
                    provider_metadata: None,
                });
            }
            parts.push(v2t::StreamPart::ReasoningStart {
                id,
                provider_metadata,
            });
        }
        parts
    }

    pub fn push_reasoning_delta(
        &mut self,
        default_reasoning_id: &str,
        delta: String,
        provider_metadata: Option<ProviderMetadata>,
    ) -> v2t::StreamPart {
        let id = self
            .reasoning_open
            .clone()
            .unwrap_or_else(|| default_reasoning_id.to_string());
        v2t::StreamPart::ReasoningDelta {
            id,
            delta,
            provider_metadata,
        }
    }

    pub fn close_reasoning(
        &mut self,
        provider_metadata: Option<ProviderMetadata>,
    ) -> Option<v2t::StreamPart> {
        self.reasoning_open
            .take()
            .map(|id| self.reasoning_end_part(id, provider_metadata))
    }

    pub fn reasoning_end_part(
        &self,
        id: String,
        provider_metadata: Option<ProviderMetadata>,
    ) -> v2t::StreamPart {
        v2t::StreamPart::ReasoningEnd {
            id,
            provider_metadata,
        }
    }

    pub fn start_tool_call(
        &mut self,
        id: String,
        name: String,
        provider_executed: bool,
        provider_metadata: Option<ProviderMetadata>,
    ) -> v2t::StreamPart {
        self.has_tool_calls = true;
        self.tool_names.insert(id.clone(), name.clone());
        v2t::StreamPart::ToolInputStart {
            id,
            tool_name: name,
            provider_executed,
            provider_metadata,
        }
    }

    pub fn push_tool_call_delta(
        &mut self,
        id: String,
        args_json: String,
        provider_executed: bool,
        provider_metadata: Option<ProviderMetadata>,
    ) -> v2t::StreamPart {
        self.tool_args
            .entry(id.clone())
            .or_default()
            .push_str(&args_json);
        v2t::StreamPart::ToolInputDelta {
            id,
            delta: args_json,
            provider_executed,
            provider_metadata,
        }
    }

    pub fn finish_tool_call(
        &mut self,
        id: String,
        provider_executed: bool,
        input_end_metadata: Option<ProviderMetadata>,
        call_metadata: Option<ProviderMetadata>,
        dynamic: bool,
        provider_options: Option<v2t::ProviderOptions>,
    ) -> Vec<v2t::StreamPart> {
        let mut parts =
            vec![self.tool_input_end_part(id.clone(), provider_executed, input_end_metadata)];
        let name = self.tool_names.remove(&id).unwrap_or_default();
        if let Some(args) = self.tool_args.remove(&id) {
            parts.push(self.tool_call_part(
                id,
                name,
                args,
                provider_executed,
                call_metadata,
                dynamic,
                provider_options,
            ));
        }
        parts
    }

    pub fn tool_input_end_part(
        &self,
        id: String,
        provider_executed: bool,
        provider_metadata: Option<ProviderMetadata>,
    ) -> v2t::StreamPart {
        v2t::StreamPart::ToolInputEnd {
            id,
            provider_executed,
            provider_metadata,
        }
    }

    pub fn tool_call_part(
        &self,
        tool_call_id: String,
        tool_name: String,
        input: String,
        provider_executed: bool,
        provider_metadata: Option<ProviderMetadata>,
        dynamic: bool,
        provider_options: Option<v2t::ProviderOptions>,
    ) -> v2t::StreamPart {
        v2t::StreamPart::ToolCall(v2t::ToolCallPart {
            tool_call_id,
            tool_name,
            input,
            provider_executed,
            provider_metadata,
            dynamic,
            provider_options,
        })
    }

    pub fn apply_usage(&mut self, usage: &crate::ai_sdk_types::TokenUsage) {
        self.usage.input_tokens = Some(usage.input_tokens as u64);
        self.usage.output_tokens = Some(usage.output_tokens as u64);
        self.usage.total_tokens = Some(usage.total_tokens as u64);
        self.usage.cached_input_tokens = usage.cache_read_tokens.map(|v| v as u64);
    }

    pub fn finish_stream(
        &mut self,
        finish: Option<(v2t::FinishReason, Option<ProviderMetadata>)>,
        finish_reason_fallback: v2t::FinishReason,
    ) -> Vec<v2t::StreamPart> {
        let mut parts = Vec::new();
        if let Some(part) = self.close_text(None) {
            parts.push(part);
        }
        if let Some(part) = self.close_reasoning(None) {
            parts.push(part);
        }
        let (finish_reason, provider_metadata) = match finish {
            Some(finish) => finish,
            None if self.has_tool_calls => (v2t::FinishReason::ToolCalls, None),
            None => (finish_reason_fallback, None),
        };
        parts.push(self.finish_part(finish_reason, provider_metadata));
        parts
    }

    pub fn finish_part(
        &self,
        finish_reason: v2t::FinishReason,
        provider_metadata: Option<ProviderMetadata>,
    ) -> v2t::StreamPart {
        v2t::StreamPart::Finish {
            usage: self.usage.clone(),
            finish_reason,
            provider_metadata,
        }
    }
}

/// Map a provider `Event` stream into provider-agnostic `StreamPart`s with
/// configurable hooks for provider metadata and finish reasoning.
pub fn map_events_to_parts<Extra, S>(stream: S, mut cfg: EventMapperConfig<Extra>) -> PartStream
where
    Extra: Send + 'static,
    S: Stream<Item = Result<ProviderEvent, SdkError>> + Send + 'static,
{
    Box::pin(try_stream! {
        let mut state = StreamNormalizationState::new(cfg.initial_extra);
        yield v2t::StreamPart::StreamStart { warnings: cfg.warnings };

        futures_util::pin_mut!(stream);
        while let Some(evt) = stream.next().await {
            match evt? {
                ProviderEvent::TextDelta { delta } => {
                    let start_metadata = if state.text_open.is_none() {
                        if let Some(f) = cfg.hooks.text_start_metadata.as_mut() {
                            f(&mut state)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    for part in state.push_text_delta(
                        None,
                        cfg.default_text_id,
                        delta,
                        start_metadata,
                        None,
                    ) {
                        yield part;
                    }
                }
                ProviderEvent::ReasoningStart { id } => {
                    let md = if let Some(f) = cfg.hooks.reasoning_start_metadata.as_mut() {
                        f(&mut state)
                    } else {
                        None
                    };
                    for part in state.open_reasoning(id, md) {
                        yield part;
                    }
                }
                ProviderEvent::ReasoningDelta { delta } => {
                    yield state.push_reasoning_delta("reasoning-1", delta, None);
                }
                ProviderEvent::ReasoningEnd => {
                    if let Some(part) = state.close_reasoning(None) {
                        yield part;
                    }
                }
                ProviderEvent::ToolCallStart { id, name } => {
                    let treat_as_text = cfg.treat_tool_names_as_text.contains(&name);
                    if treat_as_text {
                        state.tool_names.insert(id.clone(), name.clone());
                        let md = if let Some(f) = cfg.hooks.text_start_metadata.as_mut() {
                            f(&mut state)
                        } else {
                            None
                        };
                        yield v2t::StreamPart::TextStart {
                            id,
                            provider_metadata: md,
                        };
                    } else {
                        let md = if let Some(f) = cfg.hooks.tool_start_metadata.as_mut() {
                            f(&mut state, &id, &name)
                        } else {
                            None
                        };
                        yield state.start_tool_call(id, name, false, md);
                    }
                }
                ProviderEvent::ToolCallDelta { id, args_json } => {
                    let treat_as_text = state
                        .tool_names
                        .get(&id)
                        .map(|n| cfg.treat_tool_names_as_text.contains(n))
                        .unwrap_or(false);
                    if treat_as_text {
                        yield v2t::StreamPart::TextDelta {
                            id,
                            delta: args_json,
                            provider_metadata: None,
                        };
                    } else {
                        yield state.push_tool_call_delta(id, args_json, false, None);
                    }
                }
                ProviderEvent::ToolCallEnd { id } => {
                    let treat_as_text = state
                        .tool_names
                        .get(&id)
                        .map(|n| cfg.treat_tool_names_as_text.contains(n))
                        .unwrap_or(false);
                    if treat_as_text {
                        yield v2t::StreamPart::TextEnd {
                            id,
                            provider_metadata: None,
                        };
                    } else {
                        let md = if let Some(f) = cfg.hooks.tool_end_metadata.as_mut() {
                            f(&mut state, &id)
                        } else {
                            None
                        };
                        for part in state.finish_tool_call(id, false, None, md, false, None) {
                            yield part;
                        }
                    }
                }
                ProviderEvent::Usage { usage } => {
                    state.apply_usage(&usage);
                }
                ProviderEvent::Raw { raw_value } => {
                    yield v2t::StreamPart::Raw { raw_value };
                }
                ProviderEvent::Data { key, value } => {
                    if let Some(f) = cfg.hooks.data.as_mut() {
                        if let Some(extra_parts) = f(&mut state, &key, &value) {
                            for part in extra_parts {
                                yield part;
                            }
                        }
                    }
                }
                ProviderEvent::Retrying { .. } => {}
                ProviderEvent::Error { message } => {
                    yield v2t::StreamPart::Error {
                        error: serde_json::json!({"message": message}),
                    };
                }
                ProviderEvent::Done => {
                    let finish = cfg.hooks.finish.as_ref().map(|f| f(&state));
                    for part in state.finish_stream(finish, cfg.finish_reason_fallback.clone()) {
                        yield part;
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{map_events_to_parts, EventMapperConfig, EventMapperHooks};
    use crate::ai_sdk_types::v2 as v2t;
    use crate::ai_sdk_types::{Event, TokenUsage};
    use futures_util::{stream, TryStreamExt};
    use serde_json::json;
    use std::collections::HashSet;

    #[tokio::test]
    async fn extracted_state_machine_preserves_basic_stream_lifecycle() {
        let stream = stream::iter(vec![
            Ok(Event::TextDelta {
                delta: "hello".into(),
            }),
            Ok(Event::Usage {
                usage: TokenUsage {
                    input_tokens: 2,
                    output_tokens: 3,
                    total_tokens: 5,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
            }),
            Ok(Event::ToolCallStart {
                id: "tool-1".into(),
                name: "weather".into(),
            }),
            Ok(Event::ToolCallDelta {
                id: "tool-1".into(),
                args_json: "{\"city\":\"SF\"}".into(),
            }),
            Ok(Event::ToolCallEnd {
                id: "tool-1".into(),
            }),
            Ok(Event::Done),
        ]);

        let parts: Vec<v2t::StreamPart> = map_events_to_parts(
            stream,
            EventMapperConfig {
                warnings: vec![],
                treat_tool_names_as_text: HashSet::new(),
                default_text_id: "text-1",
                finish_reason_fallback: v2t::FinishReason::Stop,
                initial_extra: (),
                hooks: EventMapperHooks::default(),
            },
        )
        .try_collect()
        .await
        .expect("stream parts");

        assert_eq!(
            serde_json::to_value(&parts).expect("serialize stream parts"),
            json!([
                {
                    "type": "stream-start",
                    "warnings": []
                },
                {
                    "type": "text-start",
                    "id": "text-1"
                },
                {
                    "type": "text-delta",
                    "id": "text-1",
                    "delta": "hello"
                },
                {
                    "type": "tool-input-start",
                    "id": "tool-1",
                    "tool_name": "weather",
                    "providerExecuted": false
                },
                {
                    "type": "tool-input-delta",
                    "id": "tool-1",
                    "delta": "{\"city\":\"SF\"}",
                    "providerExecuted": false
                },
                {
                    "type": "tool-input-end",
                    "id": "tool-1",
                    "providerExecuted": false
                },
                {
                    "type": "tool-call",
                    "toolCallId": "tool-1",
                    "toolName": "weather",
                    "input": "{\"city\":\"SF\"}",
                    "providerExecuted": false
                },
                {
                    "type": "text-end",
                    "id": "text-1"
                },
                {
                    "type": "finish",
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 3,
                        "total_tokens": 5
                    },
                    "finish_reason": "tool-calls"
                }
            ])
        );
    }
}
