// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `tessera recorder ref`: the software reference recorder of RFC-0015 'Reference recorder', a
//! TCP proxy for tests, demonstrations and integration work. It is not a production recorder,
//! and does not satisfy A-ANCHOR-2 by itself.
//!
//! - **Forwarding.** Each connection to `--listen` is relayed to `--upstream` and back. In each
//!   direction, bytes are cut into frames (`tessera_anchor::recorder::FrameParser`), and each frame
//!   is forwarded once its entry is logged, so inbound frames are sequenced before the upstream
//!   can read them (RFC-0063 'Inbound before outbound'). The bytes forwarded are the bytes
//!   received, in order. Bytes that start no frame are forwarded as they come, unlogged, and the
//!   direction is not parsed further.
//! - **Sessions.** When a connection ends, the recorder logs the close of every session it saw on
//!   it.
//! - **Periods.** Every `--period` seconds it closes the period:
//!   - it writes the period's entries to `<log-dir>/<start>.tdl`, the concatenation of their
//!     TCE encodings in `seq` order (03 §5's `stream`);
//!   - it writes the signed manifest's envelope to `<log-dir>/<start>.tlm`;
//!   - it opens the next period with its carries (RFC-0062).
//!
//!   `<start>` is the period's `period_start_ns` in twenty decimal digits. Starts strictly
//!   increase along the chain, so every period has its own files, an empty one included, and
//!   the files sort in chain order. Each file is written under a temporary name and renamed into
//!   place, the log before the manifest, so a visible `.tlm` has both files complete.
//! - **The key** is the recorder's 32-byte Ed25519 seed, the whole content of `--key`.
//!
//! Everything logged is public: frames are ciphertext, and the seed stays in its `Secret`.

use std::fs;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tessera_anchor::read_seed;
use tessera_anchor::recorder::{FrameParser, Period, Recorder, encode_log, random_log_id};
use tessera_codec::error::{ErrorCode, TesseraError};
use tessera_codec::{Public, envelope, schema};

use crate::io_error;

/// The flags of `tessera recorder ref`.
pub(crate) struct Options<'a> {
    pub(crate) listen: SocketAddr,
    pub(crate) upstream: SocketAddr,
    pub(crate) log_dir: &'a Path,
    pub(crate) key: &'a Path,
    pub(crate) period: Duration,
}

/// The recorder and the directory it writes its periods to.
struct State {
    recorder: Recorder,
    log_dir: PathBuf,
}

/// The recorder's state behind its lock. A thread that panicked while holding it left a state no
/// other thread can trust, so the others stop.
fn lock(state: &Mutex<State>) -> MutexGuard<'_, State> {
    match state.lock() {
        Ok(guard) => guard,
        Err(_) => std::process::abort(),
    }
}

/// Nanoseconds since the Unix epoch, the recorder's clock.
fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Runs the recorder until the process ends.
pub(crate) fn run(options: &Options<'_>) -> Result<(), TesseraError> {
    if options.period.is_zero() {
        return Err(TesseraError::new(ErrorCode::Usage)
            .with_source("--period must be at least one second".to_owned()));
    }
    let seed = read_seed(options.key).map_err(io_error(format!(
        "cannot read the recorder seed {}",
        options.key.display()
    )))?;
    fs::create_dir_all(options.log_dir).map_err(io_error(format!(
        "cannot create {}",
        options.log_dir.display()
    )))?;
    let log_id = random_log_id().map_err(|e| TesseraError::new(e.code()))?;
    let listener = TcpListener::bind(options.listen)
        .map_err(io_error(format!("cannot listen on {}", options.listen)))?;
    let address = listener
        .local_addr()
        .map_err(io_error("cannot read the listening address".to_owned()))?;
    let state = Arc::new(Mutex::new(State {
        recorder: Recorder::new(seed, log_id, now_ns()),
        log_dir: options.log_dir.to_path_buf(),
    }));
    tessera_codec::event!(
        INFO,
        "cli.recorder_listening",
        address = Public(address),
        upstream = Public(options.upstream);
        "tessera recorder ref: listening on {address}, forwarding to {upstream}"
    );
    let rotating = Arc::clone(&state);
    let period = options.period;
    thread::spawn(move || {
        loop {
            thread::sleep(period);
            if let Err(error) = rotate(&rotating) {
                tessera_codec::event!(
                    ERROR,
                    "cli.recorder_period_failed",
                    error = Public(error);
                    "tessera recorder ref: cannot close the period: {error}"
                );
            }
        }
    });
    for client in listener.incoming() {
        let client = client.map_err(io_error("cannot accept a connection".to_owned()))?;
        let state = Arc::clone(&state);
        let upstream = options.upstream;
        thread::spawn(move || {
            if let Err(error) = relay(client, upstream, &state) {
                tessera_codec::event!(
                    WARN,
                    "cli.recorder_connection_failed",
                    error = Public(error);
                    "tessera recorder ref: a connection failed: {error}"
                );
            }
        });
    }
    Ok(())
}

/// Closes the current period and writes it out.
fn rotate(state: &Mutex<State>) -> Result<(), TesseraError> {
    let mut state = lock(state);
    let Period { manifest, entries } = state
        .recorder
        .rotate(now_ns())
        .map_err(|e| TesseraError::new(ErrorCode::Internal).with_source(e.to_string()))?;
    let stem = format!("{:020}", manifest.body.period_start_ns);
    let log = state.log_dir.join(format!("{stem}.tdl"));
    let tlm = state.log_dir.join(format!("{stem}.tlm"));
    write_whole(&log, &encode_log(&entries))?;
    write_whole(&tlm, &envelope::seal(&schema::LOG_MANIFEST, &manifest))?;
    tessera_codec::event!(
        INFO,
        "cli.recorder_period_written",
        entries = entries.len(),
        manifest = Public(tlm.display());
        "tessera recorder ref: wrote {entries} entries and {manifest}"
    );
    Ok(())
}

/// Writes `bytes` to `path` whole: to `<path>.tmp` first, then renamed into place, so a reader
/// sees the file complete or not at all. Not `write_bytes_atomically`, which also records each
/// output in the run manifest: a recorder runs until it is stopped, and would grow that list by
/// one entry a period.
fn write_whole(path: &Path, bytes: &[u8]) -> Result<(), TesseraError> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    fs::write(&tmp, bytes).map_err(io_error(format!("cannot write {}", tmp.display())))?;
    fs::rename(&tmp, path).map_err(io_error(format!("cannot write {}", path.display())))
}

/// Relays one client connection to the upstream and back, logging both directions, then logs
/// the close of every session seen on it.
fn relay(client: TcpStream, upstream: SocketAddr, state: &Arc<Mutex<State>>) -> io::Result<()> {
    let server = TcpStream::connect(upstream)?;
    let (client_in, server_out) = (client.try_clone()?, server.try_clone()?);
    let inbound = {
        let state = Arc::clone(state);
        thread::spawn(move || pump(client_in, server_out, &state))
    };
    let outbound = pump(server, client, state);
    let inbound = inbound
        .join()
        .unwrap_or_else(|_| Err(io::Error::other("the inbound relay panicked")));
    let mut sessions = inbound.as_ref().map(Vec::clone).unwrap_or_default();
    sessions.extend(outbound.as_ref().map(Vec::clone).unwrap_or_default());
    sessions.sort_unstable();
    sessions.dedup();
    let mut guard = lock(state);
    for session in sessions {
        guard.recorder.close(session, now_ns());
    }
    drop(guard);
    inbound.and(outbound).map(|_| ())
}

/// Copies one direction from `from` to `to`, logging each frame before forwarding it. Returns the
/// sessions seen.
fn pump(mut from: TcpStream, mut to: TcpStream, state: &Mutex<State>) -> io::Result<Vec<[u8; 16]>> {
    let mut parser = FrameParser::new();
    let mut parsing = true;
    let mut sessions = Vec::new();
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = from.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        if !parsing {
            to.write_all(&buffer[..read])?;
            continue;
        }
        parser.push(&buffer[..read]);
        loop {
            match parser.next_frame() {
                Ok(Some(frame)) => {
                    lock(state).recorder.record(&frame, now_ns());
                    if !sessions.contains(&frame.session_id()) {
                        sessions.push(frame.session_id());
                    }
                    to.write_all(&frame.to_bytes())?;
                }
                Ok(None) => break,
                Err(error) => {
                    tessera_codec::event!(
                        WARN,
                        "cli.recorder_unparsed",
                        error = Public(error);
                        "tessera recorder ref: forwarding the rest of a direction unlogged: {error}"
                    );
                    to.write_all(parser.pending())?;
                    parsing = false;
                    break;
                }
            }
        }
    }
    if parsing {
        to.write_all(parser.pending())?;
    }
    let _ = to.shutdown(Shutdown::Write);
    Ok(sessions)
}
