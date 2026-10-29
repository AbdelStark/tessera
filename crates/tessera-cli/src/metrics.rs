// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Opt-in, bounded Prometheus HTTP exposition of the existing registry (05 §4).
//! Timings are private-class: operators keep this listener on their side of TB-1.

use std::fmt::Write as _;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tessera_codec::metrics::{Sample, Value};

const REQUEST_LIMIT: usize = 4096;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(1);

/// Owns the listener thread; no thread or socket exists without an explicit address.
pub(crate) struct Server {
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    #[cfg(test)]
    address: SocketAddr,
}

impl Server {
    pub(crate) fn start(address: SocketAddr) -> io::Result<Self> {
        let listener = TcpListener::bind(address)?;
        listener.set_nonblocking(true)?;
        #[cfg(test)]
        let address = listener.local_addr()?;
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let worker = thread::Builder::new()
            .name("tessera-metrics".into())
            .spawn(move || {
                while !stopped.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            // A malformed or disconnected scraper cannot fail the proof,
                            // and its input/error never reaches logs (05 §3 OBS-1).
                            let _ = serve(&mut stream);
                        }
                        Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(20));
                        }
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                        Err(_) => break,
                    }
                }
            })?;
        Ok(Self {
            stop,
            worker: Some(worker),
            #[cfg(test)]
            address,
        })
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn reply(stream: &mut TcpStream, status: &str, body: &str) -> io::Result<()> {
    let allow = if status.starts_with("405") {
        "Allow: GET\r\n"
    } else {
        ""
    };
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/plain; version=0.0.4; charset=utf-8\r\nContent-Length: {}\r\n{allow}Connection: close\r\n\r\n{body}",
        body.len()
    );
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    let mut remaining = response.as_bytes();
    while !remaining.is_empty() {
        let timeout = deadline.saturating_duration_since(Instant::now());
        if timeout.is_zero() {
            return Ok(());
        }
        stream.set_write_timeout(Some(timeout))?;
        let written = stream.write(remaining)?;
        if written == 0 {
            return Ok(());
        }
        remaining = &remaining[written..];
    }
    Ok(())
}

fn serve(stream: &mut TcpStream) -> io::Result<()> {
    // Accepted sockets inherit nonblocking mode on some platforms. A fragmented
    // request must wait for its remaining headers, subject to the deadline below.
    stream.set_nonblocking(false)?;
    let deadline = Instant::now() + REQUEST_TIMEOUT;
    let mut request = [0u8; REQUEST_LIMIT];
    let mut used = 0;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(());
        }
        stream.set_read_timeout(Some(remaining))?;
        let count = stream.read(&mut request[used..])?;
        if count == 0 {
            return Ok(());
        }
        used += count;
        if request[..used].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if used == REQUEST_LIMIT {
            return reply(stream, "431 Request Header Fields Too Large", "");
        }
    }
    let Some(line) = request[..used].split(|&b| b == b'\r').next() else {
        return reply(stream, "400 Bad Request", "");
    };
    let parts: Vec<_> = line.split(|&b| b == b' ').collect();
    if parts.len() != 3 || !matches!(parts[2], b"HTTP/1.0" | b"HTTP/1.1") {
        return reply(stream, "400 Bad Request", "");
    }
    if parts[0] != b"GET" {
        return reply(stream, "405 Method Not Allowed", "");
    }
    if parts[1] != b"/metrics" {
        return reply(stream, "404 Not Found", "");
    }
    reply(
        stream,
        "200 OK",
        &render(&tessera_codec::metrics::snapshot()),
    )
}

fn number(value: f64) -> String {
    match value {
        v if v == f64::INFINITY => "+Inf".into(),
        v if v == f64::NEG_INFINITY => "-Inf".into(),
        v => v.to_string(),
    }
}

fn labels(labels: &[(&str, String)], extra: Option<(&str, &str)>) -> String {
    let mut pairs: Vec<String> = labels
        .iter()
        .map(|(key, value)| {
            let escaped = value
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n");
            format!("{key}=\"{escaped}\"")
        })
        .collect();
    if let Some((key, value)) = extra {
        pairs.push(format!("{key}=\"{value}\""));
    }
    if pairs.is_empty() {
        String::new()
    } else {
        format!("{{{}}}", pairs.join(","))
    }
}

/// Names and labels come only from the typed registry, never HTTP input (OBS-1).
fn render(samples: &[Sample]) -> String {
    let mut out = String::new();
    let mut previous = None;
    for sample in samples {
        let kind = match sample.value {
            Value::Counter(_) => "counter",
            Value::Gauge(_) => "gauge",
            Value::Histogram { .. } => "histogram",
        };
        if previous != Some(sample.name) {
            let _ = writeln!(out, "# TYPE {} {kind}", sample.name);
            previous = Some(sample.name);
        }
        let name = sample.name;
        let tags = labels(&sample.labels, None);
        match sample.value {
            Value::Counter(value) => {
                let _ = writeln!(out, "{name}{tags} {value}");
            }
            Value::Gauge(value) => {
                let _ = writeln!(out, "{name}{tags} {}", number(value));
            }
            Value::Histogram { count, sum, .. } => {
                // The registry retains aggregates, not observations. An invented
                // finite bucket would misrepresent its data (05 §4).
                let bucket = labels(&sample.labels, Some(("le", "+Inf")));
                let _ = writeln!(out, "{name}_bucket{bucket} {count}");
                let _ = writeln!(out, "{name}_sum{tags} {}", number(sum));
                let _ = writeln!(out, "{name}_count{tags} {count}");
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(server: &Server, input: &[u8]) -> String {
        let mut socket = TcpStream::connect(server.address).expect("connect");
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("timeout");
        socket.write_all(input).expect("request");
        let mut out = String::new();
        socket.read_to_string(&mut out).expect("response");
        out
    }

    #[test]
    fn exposition_parses_and_preserves_counter_gauge_and_histogram_aggregates() {
        let text = render(&[
            Sample {
                name: "tessera_count_total",
                labels: vec![("table", "7".into())],
                value: Value::Counter(23),
            },
            Sample {
                name: "tessera_memory_bytes",
                labels: vec![],
                value: Value::Gauge(17.5),
            },
            Sample {
                name: "tessera_phase_seconds",
                labels: vec![("phase", "witness".into())],
                value: Value::Histogram {
                    count: 3,
                    sum: 6.5,
                    min: 1.0,
                    max: 3.5,
                },
            },
        ]);
        let parsed = prometheus_parse::Scrape::parse(text.lines().map(|s| Ok(s.to_owned())))
            .expect("Prometheus text");
        assert!(
            parsed
                .samples
                .iter()
                .any(|s| s.metric == "tessera_count_total"
                    && s.value == prometheus_parse::Value::Counter(23.0)
                    && s.labels.get("table") == Some("7"))
        );
        assert!(
            parsed
                .samples
                .iter()
                .any(|s| s.metric == "tessera_memory_bytes"
                    && s.value == prometheus_parse::Value::Gauge(17.5))
        );
        let histogram = parsed
            .samples
            .iter()
            .find(|s| s.metric == "tessera_phase_seconds")
            .expect("histogram");
        let prometheus_parse::Value::Histogram(buckets) = &histogram.value else {
            panic!("histogram type");
        };
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[0].less_than, f64::INFINITY);
        assert_eq!(buckets[0].count, 3.0);
        assert!(text.contains("tessera_phase_seconds_sum{phase=\"witness\"} 6.5\n"));
        assert!(text.contains("tessera_phase_seconds_count{phase=\"witness\"} 3\n"));
        assert_eq!(number(f64::INFINITY), "+Inf");
        assert_eq!(number(f64::NEG_INFINITY), "-Inf");
        assert_eq!(number(f64::NAN), "NaN");
    }

    #[test]
    fn listener_serves_the_registry_and_refuses_other_requests() {
        let server = Server::start("127.0.0.1:0".parse().expect("address")).expect("listen");
        tessera_codec::metrics::lookups(987654, 23);
        let response = request(&server, b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert!(response.starts_with("HTTP/1.1 200 OK\r\n"));
        let (headers, body) = response.split_once("\r\n\r\n").expect("headers");
        assert!(headers.contains("Content-Length: "));
        assert!(body.contains("tessera_lookups_total{table=\"987654\"} 23\n"));
        prometheus_parse::Scrape::parse(body.lines().map(|s| Ok(s.to_owned())))
            .expect("parse served registry");
        for (input, status) in [
            (
                "GET /metrics?secret=canary HTTP/1.1\r\n\r\n",
                "404 Not Found",
            ),
            ("POST /metrics HTTP/1.1\r\n\r\n", "405 Method Not Allowed"),
            ("GET /metrics HTTP/2\r\n\r\n", "400 Bad Request"),
        ] {
            let response = request(&server, input.as_bytes());
            assert!(
                response.starts_with(&format!("HTTP/1.1 {status}\r\n")),
                "{response}"
            );
            assert!(!response.contains("canary"));
        }
        let oversized = vec![b'x'; 4096];
        assert!(request(&server, &oversized).starts_with("HTTP/1.1 431 "));
        assert!(
            TcpListener::bind(server.address).is_err(),
            "listener owns the explicit port"
        );
        let address = server.address;
        drop(server);
        assert!(
            TcpListener::bind(address).is_ok(),
            "drop releases the listener"
        );
    }

    #[test]
    fn fragmented_request_headers_are_read_to_completion() {
        let server = Server::start("127.0.0.1:0".parse().expect("address")).expect("listen");
        let mut socket = TcpStream::connect(server.address).expect("connect");
        socket
            .set_read_timeout(Some(Duration::from_secs(3)))
            .expect("timeout");
        socket.write_all(b"GET /met").expect("first fragment");
        thread::sleep(Duration::from_millis(80));
        socket
            .write_all(b"rics HTTP/1.1\r\n\r\n")
            .expect("second fragment");
        let mut response = String::new();
        socket.read_to_string(&mut response).expect("response");
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    }

    #[test]
    fn a_scraper_that_does_not_read_cannot_hold_a_write_forever() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listen");
        let peer = TcpStream::connect(listener.local_addr().expect("address")).expect("connect");
        let (mut stream, _) = listener.accept().expect("accept");
        let (send, receive) = std::sync::mpsc::channel();
        let writer = thread::spawn(move || {
            let body = "x".repeat(8 * 1024 * 1024);
            let _ = reply(&mut stream, "200 OK", &body);
            send.send(()).expect("notify test");
        });
        let completed = receive.recv_timeout(Duration::from_secs(3));
        // Close the peer before joining even on failure, so a planted missing
        // write timeout fails an assertion instead of hanging the test process.
        drop(peer);
        writer.join().expect("writer thread");
        assert!(
            completed.is_ok(),
            "the write must finish within its deadline"
        );
    }

    #[test]
    fn trickled_headers_do_not_restart_the_absolute_deadline() {
        let server = Server::start("127.0.0.1:0".parse().expect("address")).expect("listen");
        let mut trickle = TcpStream::connect(server.address).expect("connect");
        trickle
            .write_all(b"GET /metrics HTTP/1.1\r\nX: ")
            .expect("header prefix");
        let writer = thread::spawn(move || {
            for _ in 0..50 {
                if trickle.write_all(b"x").is_err() {
                    break;
                }
                thread::sleep(Duration::from_millis(40));
            }
        });
        thread::sleep(Duration::from_millis(80));
        let started = Instant::now();
        let response = request(&server, b"GET /metrics HTTP/1.1\r\n\r\n");
        let elapsed = started.elapsed();
        writer.join().expect("writer thread");
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(elapsed < Duration::from_millis(1700), "{elapsed:?}");
    }

    #[test]
    fn unfinished_headers_cannot_hold_the_listener_forever() {
        let server = Server::start("127.0.0.1:0".parse().expect("address")).expect("listen");
        let mut stalled = TcpStream::connect(server.address).expect("connect");
        stalled
            .write_all(b"GET /metrics HTTP/1.1\r\n")
            .expect("partial header");
        // A second request still completes once the first request's absolute deadline expires.
        assert!(request(&server, b"GET /metrics HTTP/1.1\r\n\r\n").starts_with("HTTP/1.1 200 OK"));
    }
}
