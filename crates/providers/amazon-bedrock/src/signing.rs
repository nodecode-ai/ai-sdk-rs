use std::time::SystemTime;

use crate::ai_sdk_core::error::{SdkError, TransportError};
use crate::ai_sdk_core::json::without_null_fields;
use crate::ai_sdk_core::transport::TransportConfig;
use aws_credential_types::Credentials as AwsCredentials;
use aws_sigv4::http_request::{
    sign, SignableBody, SignableRequest, SigningParams, SigningSettings,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use http::Request;
use serde_json::Value as JsonValue;

use crate::provider_amazon_bedrock::config::{BedrockAuth, SigV4Config};

#[derive(Debug, Clone)]
pub struct PreparedRequest {
    pub body: JsonValue,
    pub headers: Vec<(String, String)>,
}

pub fn prepare_request(
    auth: &BedrockAuth,
    url: &str,
    mut body: JsonValue,
    base_headers: &[(String, String)],
    transport_cfg: &TransportConfig,
) -> Result<PreparedRequest, SdkError> {
    if transport_cfg.strip_null_fields {
        body = without_null_fields(&body);
    }

    let mut headers: Vec<(String, String)> = base_headers.to_vec();
    ensure_header(&mut headers, "content-type", "application/json");
    ensure_header(&mut headers, "accept", "application/json");

    match auth {
        BedrockAuth::ApiKey { token } => {
            ensure_header(&mut headers, "authorization", &format!("Bearer {}", token));
            Ok(PreparedRequest { body, headers })
        }
        BedrockAuth::SigV4(cfg) => prepare_sigv4(cfg, url, body, headers),
    }
}

fn prepare_sigv4(
    cfg: &SigV4Config,
    url: &str,
    body: JsonValue,
    mut headers: Vec<(String, String)>,
) -> Result<PreparedRequest, SdkError> {
    let body_bytes = serde_json::to_vec(&body)?;

    ensure_host_header(url, &mut headers)?;

    let header_refs: Vec<(&str, &str)> = headers
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let signable = SignableRequest::new(
        "POST",
        url,
        header_refs.clone().into_iter(),
        SignableBody::Bytes(&body_bytes),
    )
    .map_err(|e| SdkError::Transport(TransportError::Other(e.to_string())))?;

    let mut settings = SigningSettings::default();
    settings.signature_location = aws_sigv4::http_request::SignatureLocation::Headers;

    let creds = AwsCredentials::new(
        cfg.access_key_id.clone(),
        cfg.secret_access_key.clone(),
        cfg.session_token.clone(),
        None,
        "ai-sdk-rs",
    );
    let identity = Identity::new(creds, None);
    let signing_params: SigningParams<'_> = v4::SigningParams::builder()
        .identity(&identity)
        .region(&cfg.region)
        .name("bedrock")
        .time(SystemTime::now())
        .settings(settings)
        .build()
        .map_err(|e| SdkError::Transport(TransportError::Other(e.to_string())))?
        .into();

    let signed = sign(signable, &signing_params)
        .map_err(|e| SdkError::Transport(TransportError::Other(e.to_string())))?;

    let (instructions, _) = signed.into_parts();

    let mut request_builder = Request::builder().method("POST").uri(url);
    for (k, v) in &headers {
        request_builder = request_builder.header(k.as_str(), v.as_str());
    }
    let mut request = request_builder
        .body(())
        .map_err(|e| SdkError::Transport(TransportError::Other(e.to_string())))?;
    instructions.apply_to_request_http1x(&mut request);

    let mut final_headers: Vec<(String, String)> = request
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect();

    ensure_header(&mut final_headers, "content-type", "application/json");
    ensure_header(&mut final_headers, "accept", "application/json");

    Ok(PreparedRequest {
        body,
        headers: final_headers,
    })
}

fn ensure_header(headers: &mut Vec<(String, String)>, key: &str, value: &str) {
    if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(key)) {
        headers.push((key.to_string(), value.to_string()))
    }
}

fn ensure_host_header(url: &str, headers: &mut Vec<(String, String)>) -> Result<(), SdkError> {
    if headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("host")) {
        return Ok(());
    }
    let parsed = url::Url::parse(url)
        .map_err(|e| SdkError::Transport(TransportError::Other(e.to_string())))?;
    if let Some(host) = parsed.host_str() {
        headers.push(("host".to_string(), host.to_string()));
        Ok(())
    } else {
        Err(SdkError::Transport(TransportError::Other(
            "Unable to determine host for SigV4 signing".to_string(),
        )))
    }
}
