use std::collections::{BTreeMap, HashMap};
use std::time::SystemTime;

use crate::ai_sdk_core::image::{ImageModel, ImageResponse, ImageResponseMeta};
use crate::ai_sdk_core::options::is_internal_sdk_header;
use crate::ai_sdk_core::transport::{HttpTransport, MultipartForm, TransportConfig};
use crate::ai_sdk_core::SdkError;
use crate::ai_sdk_types::image::{ImageData, ImageFile, ImageOptions, ImageUsage, ImageWarning};
use crate::ai_sdk_types::v2 as v2t;
use serde::Deserialize;
use serde_json::{json, Value as JsonValue};

use crate::provider_openai_compatible::error::map_transport_error_to_sdk_error;
use crate::provider_openai_compatible::image::options::{
    apply_provider_defaults, parse_openai_compatible_image_provider_options,
    OpenAICompatibleImageProviderOptions,
};

pub struct OpenAICompatibleImageConfig<T: HttpTransport> {
    pub provider_scope_name: String,
    pub base_url: String,
    pub headers: Vec<(String, String)>,
    pub http: T,
    pub transport_cfg: TransportConfig,
    pub query_params: Vec<(String, String)>,
    pub default_options: Option<v2t::ProviderOptions>,
}

pub struct OpenAICompatibleImageModel<T: HttpTransport = crate::reqwest_transport::ReqwestTransport>
{
    model_id: String,
    cfg: OpenAICompatibleImageConfig<T>,
}

impl<T: HttpTransport> OpenAICompatibleImageModel<T> {
    pub fn new(model_id: impl Into<String>, cfg: OpenAICompatibleImageConfig<T>) -> Self {
        Self {
            model_id: model_id.into(),
            cfg,
        }
    }

    fn build_request_url(&self, path: &str) -> String {
        let base = self.cfg.base_url.trim_end_matches('/');
        let mut url = format!("{base}{path}");
        if !self.cfg.query_params.is_empty() {
            let qp = self
                .cfg
                .query_params
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding::encode(k), urlencoding::encode(v)))
                .collect::<Vec<_>>()
                .join("&");
            url.push('?');
            url.push_str(&qp);
        }
        url
    }

    fn canonicalize_header(lc: &str) -> String {
        lc.split('-')
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    None => String::new(),
                    Some(f) => {
                        f.to_ascii_uppercase().to_string() + &chars.as_str().to_ascii_lowercase()
                    }
                }
            })
            .collect::<Vec<_>>()
            .join("-")
    }

    fn build_headers(
        &self,
        extra: &HashMap<String, String>,
        include_content_type: bool,
    ) -> Vec<(String, String)> {
        let mut hdrs: BTreeMap<String, String> = BTreeMap::new();
        for (k, v) in &self.cfg.headers {
            if is_internal_sdk_header(k) {
                continue;
            }
            let kl = k.to_ascii_lowercase();
            if !include_content_type && kl == "content-type" {
                continue;
            }
            hdrs.insert(kl, v.clone());
        }
        for (k, v) in extra {
            if is_internal_sdk_header(k) {
                continue;
            }
            let kl = k.to_ascii_lowercase();
            if !include_content_type && kl == "content-type" {
                continue;
            }
            hdrs.insert(kl, v.clone());
        }
        if include_content_type {
            hdrs.entry("content-type".into())
                .or_insert_with(|| "application/json".into());
        }
        hdrs.entry("accept".into())
            .or_insert_with(|| "application/json".into());
        hdrs.into_iter()
            .map(|(k, v)| (Self::canonicalize_header(&k), v))
            .collect()
    }

    fn headers_vec_to_map(headers: Vec<(String, String)>) -> Option<v2t::Headers> {
        if headers.is_empty() {
            return None;
        }
        Some(
            headers
                .into_iter()
                .map(|(k, v)| (k.to_ascii_lowercase(), v))
                .collect(),
        )
    }

    fn warnings_for_options(&self, options: &ImageOptions) -> Vec<ImageWarning> {
        let mut warnings = Vec::new();
        if options.aspect_ratio.is_some() {
            warnings.push(ImageWarning::Unsupported {
                feature: "aspectRatio".into(),
                details: Some(
                    "This model does not support aspect ratio. Use `size` instead.".into(),
                ),
            });
        }
        if options.seed.is_some() {
            warnings.push(ImageWarning::Unsupported {
                feature: "seed".into(),
                details: None,
            });
        }
        warnings
    }

    fn build_generation_body(&self, options: &ImageOptions) -> Result<JsonValue, SdkError> {
        let (prov_opts, prov_extras) =
            parse_openai_compatible_image_provider_options(&options.provider_options);

        let OpenAICompatibleImageProviderOptions { user } = prov_opts;

        let mut body_map = serde_json::Map::new();
        body_map.insert("model".into(), json!(self.model_id));
        if let Some(prompt) = &options.prompt {
            body_map.insert("prompt".into(), json!(prompt));
        }
        body_map.insert("n".into(), json!(options.n));
        if let Some(size) = &options.size {
            body_map.insert("size".into(), json!(size));
        }
        if let Some(user) = user {
            body_map.insert("user".into(), json!(user));
        }

        if let Some(extras) = prov_extras {
            for (k, v) in extras {
                body_map.insert(k, v);
            }
        }

        body_map.insert("response_format".into(), json!("b64_json"));
        Ok(JsonValue::Object(body_map))
    }

    fn image_data_to_bytes(data: &ImageData) -> Result<Vec<u8>, SdkError> {
        match data {
            ImageData::Base64(b64) => {
                use base64::engine::general_purpose::STANDARD as B64;
                use base64::Engine;
                B64.decode(b64.as_bytes()).map_err(|_| SdkError::Upstream {
                    status: 400,
                    message: "invalid base64 image data".into(),
                    source: None,
                })
            }
            ImageData::Bytes { bytes } => Ok(bytes.clone()),
        }
    }

    fn filename_for_media(base: &str, index: Option<usize>, media_type: &str) -> String {
        let mut name = base.to_string();
        if let Some(idx) = index {
            name.push('-');
            name.push_str(&idx.to_string());
        }
        if let Some(ext) = media_type.split('/').nth(1).filter(|s| !s.is_empty()) {
            name.push('.');
            name.push_str(ext);
        }
        name
    }

    async fn file_to_form_part(
        &self,
        file: &ImageFile,
        base_name: &str,
        index: Option<usize>,
    ) -> Result<(Vec<u8>, Option<String>, Option<String>), SdkError> {
        match file {
            ImageFile::File { media_type, data } => {
                let bytes = Self::image_data_to_bytes(data)?;
                let filename = Some(Self::filename_for_media(base_name, index, media_type));
                Ok((bytes, filename, Some(media_type.clone())))
            }
            ImageFile::Url { url } => {
                let (bytes, headers) = self
                    .cfg
                    .http
                    .get_bytes(url, &[], &self.cfg.transport_cfg)
                    .await
                    .map_err(SdkError::from)?;
                let content_type = headers
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case("content-type"))
                    .map(|(_, v)| v.clone());
                Ok((bytes.to_vec(), None, content_type))
            }
        }
    }

    fn append_provider_extras_to_form(
        form: &mut MultipartForm,
        extras: Option<serde_json::Map<String, JsonValue>>,
    ) {
        if let Some(extras) = extras {
            for (k, v) in extras {
                let value = match v {
                    JsonValue::String(s) => s,
                    other => other.to_string(),
                };
                form.push_text(k, value);
            }
        }
    }

    async fn build_edit_form(&self, options: &ImageOptions) -> Result<MultipartForm, SdkError> {
        let (prov_opts, prov_extras) =
            parse_openai_compatible_image_provider_options(&options.provider_options);

        let OpenAICompatibleImageProviderOptions { user } = prov_opts;

        let mut form = MultipartForm::new();
        form.push_text("model", self.model_id.clone());
        if let Some(prompt) = &options.prompt {
            form.push_text("prompt", prompt.clone());
        }
        form.push_text("n", options.n.to_string());
        if let Some(size) = &options.size {
            form.push_text("size", size.clone());
        }
        if let Some(user) = user {
            form.push_text("user", user);
        }

        for (idx, file) in options.files.iter().enumerate() {
            let (bytes, filename, content_type) =
                self.file_to_form_part(file, "image", Some(idx)).await?;
            form.push_bytes("image", bytes, filename, content_type);
        }

        if let Some(mask) = &options.mask {
            let (bytes, filename, content_type) =
                self.file_to_form_part(mask, "mask", None).await?;
            form.push_bytes("mask", bytes, filename, content_type);
        }

        Self::append_provider_extras_to_form(&mut form, prov_extras);
        Ok(form)
    }

    async fn handle_response(
        &self,
        json: JsonValue,
        headers: Vec<(String, String)>,
        warnings: Vec<ImageWarning>,
        request_body: Option<JsonValue>,
    ) -> Result<ImageResponse, SdkError> {
        let parsed: OpenAICompatibleImageResponse =
            serde_json::from_value(json.clone()).map_err(SdkError::Serde)?;
        let images = parsed
            .data
            .into_iter()
            .map(|item| ImageData::Base64(item.b64_json))
            .collect::<Vec<_>>();

        Ok(ImageResponse {
            images,
            warnings,
            provider_metadata: parsed.provider_metadata,
            response: ImageResponseMeta {
                timestamp: SystemTime::now(),
                model_id: self.model_id.clone(),
                headers: Self::headers_vec_to_map(headers),
            },
            usage: parsed.usage,
            response_body: Some(json),
            request_body,
        })
    }
}

#[async_trait::async_trait]
impl<T: HttpTransport + Send + Sync> ImageModel for OpenAICompatibleImageModel<T> {
    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn max_images_per_call(&self) -> Option<usize> {
        Some(10)
    }

    async fn do_generate(&self, options: ImageOptions) -> Result<ImageResponse, SdkError> {
        let scope_name = if self.cfg.provider_scope_name == "openai-compatible" {
            "openai"
        } else {
            &self.cfg.provider_scope_name
        };
        let options =
            apply_provider_defaults(options, scope_name, self.cfg.default_options.as_ref());

        let warnings = self.warnings_for_options(&options);
        let has_files = !options.files.is_empty();

        if has_files {
            let form = self.build_edit_form(&options).await?;
            let headers = self.build_headers(&options.headers, false);
            let url = self.build_request_url("/images/edits");
            let (json, res_headers) = self
                .cfg
                .http
                .post_multipart(&url, &headers, &form, &self.cfg.transport_cfg)
                .await
                .map_err(map_transport_error_to_sdk_error)?;
            return self
                .handle_response(json, res_headers, warnings, None)
                .await;
        }

        let body = self.build_generation_body(&options)?;
        let headers = self.build_headers(&options.headers, true);
        let url = self.build_request_url("/images/generations");
        let (json, res_headers) = self
            .cfg
            .http
            .post_json(&url, &headers, &body, &self.cfg.transport_cfg)
            .await
            .map_err(map_transport_error_to_sdk_error)?;

        self.handle_response(json, res_headers, warnings, Some(body))
            .await
    }
}

#[derive(Debug, Deserialize)]
struct OpenAICompatibleImageResponse {
    data: Vec<OpenAICompatibleImageData>,
    #[serde(default)]
    usage: Option<ImageUsage>,
    #[serde(default, rename = "providerMetadata")]
    provider_metadata: Option<v2t::ProviderMetadata>,
}

#[derive(Debug, Deserialize)]
struct OpenAICompatibleImageData {
    #[serde(rename = "b64_json")]
    b64_json: String,
}
