// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Explicit, host-owned OTLP/HTTP JSON trace export (05 §5, OBS-1, TB-1).
//! No environment variable enables export. Resource attributes come only from
//! this module; dependency diagnostics do not enter the export layer.

use std::time::Duration;

use opentelemetry_otlp::{Protocol, RetryPolicy, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::{
    BatchConfigBuilder, BatchSpanProcessor, Sampler, SdkTracerProvider, SpanLimits,
};
use tessera_codec::error::{ErrorCode, TesseraError};

pub(crate) fn start(endpoint: &str) -> Result<SdkTracerProvider, TesseraError> {
    let invalid = || TesseraError::new(ErrorCode::Usage);
    let url = reqwest::Url::parse(endpoint).map_err(|_| invalid())?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err(invalid());
    }
    let timeout = Duration::from_secs(1);
    let client = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .connect_timeout(timeout)
        .redirect(reqwest::redirect::Policy::none())
        .no_proxy()
        .build()
        .map_err(|_| TesseraError::new(ErrorCode::ResIo))?;
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpJson)
        .with_endpoint(url.as_str())
        .with_timeout(timeout)
        .with_http_client(client)
        .with_retry_policy(RetryPolicy::default().with_max_retries(0))
        .with_max_request_body_size(1024 * 1024)
        .build()
        .map_err(|_| TesseraError::new(ErrorCode::ResIo))?;
    let processor = BatchSpanProcessor::builder(exporter)
        .with_batch_config(
            BatchConfigBuilder::default()
                .with_max_queue_size(512)
                .with_max_export_batch_size(64)
                .with_scheduled_delay(Duration::from_secs(1))
                .build(),
        )
        .build();
    Ok(SdkTracerProvider::builder()
        .with_span_processor(processor)
        .with_sampler(Sampler::AlwaysOn)
        .with_span_limits(SpanLimits::default())
        .with_resource(
            Resource::builder_empty()
                .with_service_name("tessera")
                .build(),
        )
        .build())
}
