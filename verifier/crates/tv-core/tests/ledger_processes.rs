// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The ledger's file where another process takes part (RFC-0067 'The path', 'One writer'): a
//! FIFO made by `mkfifo`, and a writer in a child process, this binary run again.
//!
//! These tests spawn processes, so they live in a test binary of their own and run one at a
//! time. A child holds a copy of every open file description of its parent until it executes
//! its program, a directory's lock among them: a writer dropped by another test meanwhile would
//! hold its lock a while longer, and that test's next writer would be refused.

#![cfg(unix)]

use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError, mpsc};
use std::time::Duration;

use tv_core::audit::{LedgerFile, read_ledger};
use tv_core::verdict::{ErrorCode as Code, VerifierError};

const IO: VerifierError = VerifierError { code: Code::Io };
const USAGE: VerifierError = VerifierError { code: Code::Usage };

/// One test at a time: no process is spawned while another test's writer drops its lock.
fn serial() -> MutexGuard<'static, ()> {
    static SERIAL: Mutex<()> = Mutex::new(());
    SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
}

fn scratch(name: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "tessera-ledger-{name}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    fs::canonicalize(dir).unwrap()
}

#[test]
fn a_fifo_is_refused_before_it_is_opened() {
    let _serial = serial();
    let dir = scratch("fifo");
    let fifo = dir.join("ledger.tvl");
    let made = Command::new("mkfifo").arg(&fifo).status().unwrap();
    assert!(made.success());
    // Opening a FIFO for reading blocks until a writer comes: the reader must not open it, nor a
    // writer that takes it for the ledger's directory, which it opens to lock.
    let (send, receive) = mpsc::channel();
    let path = fifo.clone();
    std::thread::spawn(move || {
        let read = read_ledger(&path).err();
        let open = LedgerFile::open(&path).err();
        let inside = path.join("ledger.tvl");
        let as_dir = (
            LedgerFile::open(&inside).err(),
            LedgerFile::create(&inside).err(),
        );
        let _ = send.send((read, open, as_dir));
    });
    let (read, open, as_dir) = receive
        .recv_timeout(Duration::from_secs(20))
        .expect("the FIFO was opened and blocked");
    assert_eq!((read, open), (Some(USAGE), Some(USAGE)));
    assert_eq!(as_dir, (Some(IO), Some(IO)));
    // The refusal released the lock: a writer takes the directory at once.
    drop(LedgerFile::create(&dir.join("other.tvl")).unwrap());
    fs::remove_file(dir.join("other.tvl")).unwrap();
    fs::remove_file(&fifo).unwrap();
    fs::remove_dir(&dir).unwrap();
}

/// The environment variable that makes [`a_child_process_holds_the_ledger`] hold a ledger.
const HOLDER: &str = "TESSERA_TEST_LEDGER_HOLDER";
/// The line the child prints once it holds the ledger.
const HELD: &str = "tessera-test: the ledger is held";

/// The child process of [`a_second_process_is_refused_and_a_crash_releases_the_lock`]: it opens
/// the ledger its parent names, says so, and holds it until its standard input closes or it is
/// killed. Run without its parent's variable, it does nothing.
#[test]
#[ignore = "a child process of a_second_process_is_refused_and_a_crash_releases_the_lock"]
fn a_child_process_holds_the_ledger() {
    let Some(path) = std::env::var_os(HOLDER) else {
        return;
    };
    let file = LedgerFile::open(Path::new(&path)).unwrap();
    let mut out = io::stdout();
    writeln!(out, "{HELD}").unwrap();
    out.flush().unwrap();
    let _ = io::stdin().read_to_end(&mut Vec::new());
    drop(file);
}

/// RFC-0067 'The lock', 'No stale lock', across processes: a child process holds the ledger;
/// this process is refused with `ERR-RES-IO`, by the ledger's path, by another ledger's in the
/// directory and, once the directory is renamed, by its new name, while it still reads the
/// ledger. The child is killed, a crash, which runs no destructor: the system releases the lock,
/// and this process holds the ledger with no operator step.
#[test]
fn a_second_process_is_refused_and_a_crash_releases_the_lock() {
    let _serial = serial();
    let dir = scratch("process");
    let path = dir.join("ledger.tvl");
    drop(LedgerFile::create(&path).unwrap());
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "a_child_process_holds_the_ledger",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(HOLDER, &path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    // The child's output, line by line, until it holds the ledger; libtest may print the test's
    // name on the same line, before the child's own words.
    let stdout = child.stdout.take().unwrap();
    let (send, receive) = mpsc::channel();
    std::thread::spawn(move || {
        let mut seen = Vec::new();
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if line.ends_with(HELD) {
                let _ = send.send(Ok(()));
                return;
            }
            seen.push(line);
        }
        let _ = send.send(Err(seen));
    });
    match receive.recv_timeout(Duration::from_secs(60)) {
        Ok(Ok(())) => {}
        other => {
            let _ = child.kill();
            panic!("the child did not hold the ledger: {other:?}");
        }
    }
    // Held by the child: refused here, however the ledger is reached, and read all the same.
    assert!(read_ledger(&path).is_ok());
    assert_eq!(LedgerFile::open(&path).err(), Some(IO));
    let other = dir.join("other.tvl");
    assert_eq!(LedgerFile::create(&other).err(), Some(IO));
    assert!(!other.exists());
    let renamed = dir.with_extension("renamed");
    fs::rename(&dir, &renamed).unwrap();
    assert!(read_ledger(&renamed.join("ledger.tvl")).is_ok());
    assert_eq!(
        LedgerFile::open(&renamed.join("ledger.tvl")).err(),
        Some(IO)
    );
    fs::rename(&renamed, &dir).unwrap();
    // The crash: SIGKILL, no destructor runs, and the system releases the lock.
    child.kill().unwrap();
    let status = child.wait().unwrap();
    assert_eq!(status.signal(), Some(9), "{status:?}");
    let file = LedgerFile::open(&path).unwrap();
    // Held here now, against another writer.
    assert_eq!(LedgerFile::open(&path).err(), Some(IO));
    drop(file);
    assert_eq!(fs::read_dir(&dir).unwrap().count(), 1, "the ledger alone");
    fs::remove_file(&path).unwrap();
    fs::remove_dir(&dir).unwrap();
}
