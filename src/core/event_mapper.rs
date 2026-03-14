use std::collections::{HashMap, HashSet};

use crate::ai_sdk_types::v2 as v2t;
use crate::ai_sdk_types::Event as ProviderEvent;
use async_stream::try_stream;
use futures_core::Stream;
use futures_util::StreamExt;

use crate::core::error::SdkError;
use crate::core::v2::PartStream;

pub type ProviderMetadata = HashMap<String, HashMap<String, serde_json::Value>>;

type MetaFn<Extra> =
    Box<dyn FnMut(&mut EventMapperState<Extra>) -> Option<ProviderMetadata> + Send>;
type ToolMetaFn<Extra> =
    Box<dyn FnMut(&mut EventMapperState<Extra>, &str, &str) -> Option<ProviderMetadata> + Send>;
type ToolEndMetaFn<Extra> =
    Box<dyn FnMut(&mut EventMapperState<Extra>, &str) -> Option<ProviderMetadata> + Send>;
type DataFn<Extra> = Box<
    dyn FnMut(
            &mut EventMapperState<Extra>,
            &str,
            &serde_json::Value,
        ) -> Option<Vec<v2t::StreamPart>>
        + Send,
>;
type FinishFn<Extra> =
    Box<dyn Fn(&EventMapperState<Extra>) -> (v2t::FinishReason, Option<ProviderMetadata>) + Send>;

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

pub struct EventMapperState<Extra> {
    pub text_open: Option<String>,
    pub reasoning_open: Option<String>,
    pub tool_args: HashMap<String, String>,
    pub tool_names: HashMap<String, String>,
    pub usage: v2t::Usage,
    pub has_tool_calls: bool,
    pub extra: Extra,
}

impl<Extra> EventMapperState<Extra> {
    fn new(extra: Extra) -> Self {
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
}

/// Map a provider `Event` stream into provider-agnostic `StreamPart`s with
/// configurable hooks for provider metadata and finish reasoning.
pub fn map_events_to_parts<Extra, S>(stream: S, mut cfg: EventMapperConfig<Extra>) -> PartStream
where
    Extra: Send + 'static,
    S: Stream<Item = Result<ProviderEvent, SdkError>> + Send + 'static,
{
    Box::pin(try_stream! {
        let mut state = EventMapperState::new(cfg.initial_extra);
        yield v2t::StreamPart::StreamStart { warnings: cfg.warnings };

        futures_util::pin_mut!(stream);
        while let Some(evt) = stream.next().await {
            match evt? {
                ProviderEvent::TextDelta { delta } => {
                    let id = if let Some(id) = &state.text_open {
                        id.clone()
                    } else {
                        let id = cfg.default_text_id.to_string();
                        if let Some(f) = cfg.hooks.text_start_metadata.as_mut() {
                            let md = f(&mut state);
                            yield v2t::StreamPart::TextStart { id: id.clone(), provider_metadata: md };
                        } else {
                            yield v2t::StreamPart::TextStart { id: id.clone(), provider_metadata: None };
                        }
                        state.text_open = Some(id.clone());
                        id
                    };
                    yield v2t::StreamPart::TextDelta { id, delta, provider_metadata: None };
                }
                ProviderEvent::ReasoningStart { id } => {
                    state.reasoning_open = Some(id.clone());
                    let md = if let Some(f) = cfg.hooks.reasoning_start_metadata.as_mut() {
                        f(&mut state)
                    } else {
                        None
                    };
                    yield v2t::StreamPart::ReasoningStart { id, provider_metadata: md };
                }
                ProviderEvent::ReasoningDelta { delta } => {
                    let id = state.reasoning_open.clone().unwrap_or_else(|| "reasoning-1".into());
                    yield v2t::StreamPart::ReasoningDelta { id, delta, provider_metadata: None };
                }
                ProviderEvent::ReasoningEnd => {
                    if let Some(id) = state.reasoning_open.take() {
                        yield v2t::StreamPart::ReasoningEnd { id, provider_metadata: None };
                    }
                }
                ProviderEvent::ToolCallStart { id, name } => {
                    state.has_tool_calls = true;
                    let treat_as_text = cfg.treat_tool_names_as_text.contains(&name);
                    state.tool_names.insert(id.clone(), name.clone());
                    if treat_as_text {
                        let md = if let Some(f) = cfg.hooks.text_start_metadata.as_mut() {
                            f(&mut state)
                        } else {
                            None
                        };
                        yield v2t::StreamPart::TextStart { id, provider_metadata: md };
                    } else {
                        let md = if let Some(f) = cfg.hooks.tool_start_metadata.as_mut() {
                            f(&mut state, &id, &name)
                        } else {
                            None
                        };
                        yield v2t::StreamPart::ToolInputStart {
                            id,
                            tool_name: name,
                            provider_executed: false,
                            provider_metadata: md,
                        };
                    }
                }
                ProviderEvent::ToolCallDelta { id, args_json } => {
                    let _ = state.tool_args.entry(id.clone()).or_insert_with(String::new).push_str(&args_json);
                    let treat_as_text = state
                        .tool_names
                        .get(&id)
                        .map(|n| cfg.treat_tool_names_as_text.contains(n))
                        .unwrap_or(false);
                    if treat_as_text {
                        yield v2t::StreamPart::TextDelta { id, delta: args_json, provider_metadata: None };
                    } else {
                        yield v2t::StreamPart::ToolInputDelta {
                            id,
                            delta: args_json,
                            provider_executed: false,
                            provider_metadata: None,
                        };
                    }
                }
                ProviderEvent::ToolCallEnd { id } => {
                    let treat_as_text = state
                        .tool_names
                        .get(&id)
                        .map(|n| cfg.treat_tool_names_as_text.contains(n))
                        .unwrap_or(false);
                    if treat_as_text {
                        yield v2t::StreamPart::TextEnd { id, provider_metadata: None };
                    } else {
                        yield v2t::StreamPart::ToolInputEnd {
                            id: id.clone(),
                            provider_executed: false,
                            provider_metadata: None,
                        };
                        let md = if let Some(f) = cfg.hooks.tool_end_metadata.as_mut() {
                            f(&mut state, &id)
                        } else {
                            None
                        };
                        if let Some(args) = state.tool_args.remove(&id) {
                            let name = state.tool_names.remove(&id).unwrap_or_default();
                            yield v2t::StreamPart::ToolCall(v2t::ToolCallPart {
                                tool_call_id: id,
                                tool_name: name,
                                input: args,
                                provider_executed: false,
                                provider_metadata: md,
                                dynamic: false,
                                provider_options: None,
                            });
                        }
                    }
                }
                ProviderEvent::Usage { usage } => {
                    state.usage.input_tokens = Some(usage.input_tokens as u64);
                    state.usage.output_tokens = Some(usage.output_tokens as u64);
                    state.usage.total_tokens = Some(usage.total_tokens as u64);
                    state.usage.cached_input_tokens = usage.cache_read_tokens.map(|v| v as u64);
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
                    yield v2t::StreamPart::Error { error: serde_json::json!({"message": message}) };
                }
                ProviderEvent::Done => {
                    if let Some(id) = state.text_open.take() {
                        yield v2t::StreamPart::TextEnd { id, provider_metadata: None };
                    }
                    if let Some(id) = state.reasoning_open.take() {
                        yield v2t::StreamPart::ReasoningEnd { id, provider_metadata: None };
                    }
                    let (finish_reason, provider_metadata) = if let Some(f) = cfg.hooks.finish.as_ref() {
                        f(&state)
                    } else if state.has_tool_calls {
                        (v2t::FinishReason::ToolCalls, None)
                    } else {
                        (cfg.finish_reason_fallback.clone(), None)
                    };
                    yield v2t::StreamPart::Finish {
                        usage: state.usage.clone(),
                        finish_reason,
                        provider_metadata,
                    };
                }
            }
        }
    })
}
