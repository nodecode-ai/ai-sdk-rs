use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::v2::{headers_is_empty, provider_options_is_empty, ProviderOptions};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ImageData {
    Base64(String),
    Bytes {
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageFile {
    File {
        #[serde(rename = "mediaType")]
        media_type: String,
        data: ImageData,
    },
    Url {
        url: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageWarning {
    Unsupported {
        feature: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    Compatibility {
        feature: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<String>,
    },
    Other {
        message: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ImageUsage {
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "inputTokens"
    )]
    pub input_tokens: Option<u64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "outputTokens"
    )]
    pub output_tokens: Option<u64>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "totalTokens"
    )]
    pub total_tokens: Option<u64>,
}

fn default_image_count() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default = "default_image_count")]
    pub n: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        rename = "aspectRatio"
    )]
    pub aspect_ratio: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<ImageFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<ImageFile>,
    #[serde(default, skip_serializing_if = "headers_is_empty")]
    pub headers: HashMap<String, String>,
    #[serde(
        default,
        skip_serializing_if = "provider_options_is_empty",
        rename = "providerOptions"
    )]
    pub provider_options: ProviderOptions,
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            prompt: None,
            n: default_image_count(),
            size: None,
            aspect_ratio: None,
            seed: None,
            files: Vec::new(),
            mask: None,
            headers: HashMap::new(),
            provider_options: ProviderOptions::new(),
        }
    }
}

impl ImageOptions {
    pub fn new(prompt: Option<String>) -> Self {
        Self {
            prompt,
            ..Default::default()
        }
    }
}
