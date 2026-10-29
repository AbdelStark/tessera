# Optional prover-side telemetry

The prover CLI collects the [05 §4](spec/05-observability.md#4-metrics)
registry in-process and writes its samples to a private run manifest.
No listener or exporter starts by default, including when `OTEL_*` variables
are set. These options belong to `tessera`; the independent verifier gains
no telemetry dependency.

## Prometheus

Pass `--metrics-addr IP:PORT` to expose `GET /metrics` while a command runs:

```sh
tessera recorder ref --listen 127.0.0.1:8000 --upstream 127.0.0.1:8001 \
  --log-dir ./logs --key recorder.key --period 60 \
  --metrics-addr 127.0.0.1:9090
curl http://127.0.0.1:9090/metrics
```

The endpoint uses Prometheus text format 0.0.4. Names and labels come from the
typed registry, with no request data reflected into the response. Histograms
expose their count, sum and `+Inf` bucket: the registry retains no finite bucket
distribution. Only metrics recorded by that command appear; starting a recorder
does not invent prover measurements.

The listener has no authentication or TLS. Keep access on the prover's private
side of TB-1: phase timings and derived unit costs depend on secret inputs
([05 §4](spec/05-observability.md#4-metrics)). A loopback address suits a local
scraper; other bind addresses require operator-controlled network access.
An occupied or unavailable address fails with `ERR-RES-IO` before dispatch.
Requests have a 4 KiB header limit, a one-second total read deadline and
a one-second total write deadline. Connections close after one response; one
scrape runs at a time. Malformed, stalled or disconnected scrapers cannot fail
the proof. The listener stops when the command returns.

## OTLP traces

```sh
cargo build --locked -p tessera-cli --features otel
tessera compile --model MODEL.d --params ni-100 --out-dir compiled \
  --otlp-endpoint https://collector.example/v1/traces --log-level debug
```

`--otlp-endpoint` accepts the full trace endpoint URL; it does not append a path.
The flag exists only with `otel`. Both the feature and explicit flag are
required to export. The wire format is OTLP/HTTP JSON, not gRPC. HTTPS verifies
the collector through system TLS trust; Linux builds need the OpenSSL development
package and `pkg-config`. HTTP is available for a local collector. The client
follows no redirect and uses no automatic proxy. URLs with credentials, queries
or fragments are refused with `ERR-USAGE`, without echoing the URL.

Only Tessera's typed public-data spans/events enter the exporter (OBS-1).
`--log-level` also selects exported detail. The resource contains only the fixed
service name `tessera`; environment resource attributes and `OTEL_SERVICE_NAME`
are not included. OpenTelemetry's `OTEL_EXPORTER_OTLP_TRACES_HEADERS` (or its
general `OTEL_EXPORTER_OTLP_HEADERS` fallback) can supply collector authentication
headers after explicit opt-in. Neither variable enables export by itself.

The exporter queues at most 512 completed spans, exports at most 64 per batch,
caps a request body at 1 MiB, uses a one-second HTTP timeout and does not retry.
Shutdown attempts to flush for two seconds. Delivery is best effort: a full
queue, unreachable collector or failed export can lose diagnostics and does not
change the command's result. Configuration errors fail before dispatch; their
messages include no exporter error payload. The run manifest records `otel`
when the binary enables that feature.

## Verification

```sh
cargo test -p tessera-cli --lib --test telemetry_export --test logging --locked
cargo test -p tessera-cli --lib --test telemetry_export --test logging --features otel --locked
```

Tests parse served text with an independent Prometheus parser, receive real OTLP
JSON spans, check resource canaries and explicit endpoint precedence, refuse
redirects, and exercise malformed, fragmented and stalled HTTP requests. Both CI
variants compile a tiny model in an isolated network namespace under `strace`,
with exporter environment variables set and no opt-in flags. Any network syscall
fails that check. A connection-attempt negative control proves the tracer detects
an attempt even when isolation refuses it.
