// SPDX-License-Identifier: Apache-2.0 OR MIT

//! `envelope::write_file` (06 §7, RFC-0040 'Writing'): an artifact is written by atomic rename,
//! a secret-class one with mode 0600 from the moment its file exists, and a write that stops
//! part way leaves the previous file whole.

#![cfg(unix)]

use std::cell::Cell;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use tessera_codec::envelope::{self, write_file};
use tessera_codec::error::SchemaId;
use tessera_codec::schema::{Class, Schema};
use tessera_codec::{Encode, Sink};

/// A directory of its own for each test, under Cargo's scratch directory.
fn directory(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("files-{name}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

fn schema(class: Class) -> Schema {
    Schema {
        name: "FileProbe",
        id: SchemaId(0xFFF1),
        version: 1,
        extension: None,
        class,
    }
}

fn mode(path: &Path) -> u32 {
    fs::metadata(path).expect("metadata").permissions().mode() & 0o777
}

/// The names in `dir`, sorted.
fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .expect("a directory")
        .map(|entry| {
            entry
                .expect("an entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

#[test]
fn a_secret_artifact_has_mode_0600_and_a_public_one_the_default_mode() {
    let dir = directory("modes");
    // The default mode is what a file created without one gets, under this process's umask.
    let plain = dir.join("plain");
    fs::write(&plain, b"x").expect("a plain file");
    let default = mode(&plain);
    assert_ne!(
        default, 0o600,
        "the test needs a umask that leaves group or other bits"
    );

    let value = vec![1u32, 2, 3];
    for (class, expected) in [
        (Class::Secret, 0o600),
        (Class::Public, default),
        (Class::Private, default),
        (Class::Protocol, default),
    ] {
        let path = dir.join(format!("{class:?}.bin"));
        let digest = write_file(&path, &schema(class), &value).expect("written");
        let bytes = fs::read(&path).expect("the file");
        assert_eq!(bytes, envelope::seal(&schema(class), &value), "{class:?}");
        assert_eq!(
            &bytes[bytes.len() - 32..],
            digest,
            "{class:?}: the digest returned"
        );
        assert_eq!(mode(&path), expected, "{class:?}");
    }
    // A secret artifact replaces a file of another mode with one of its own.
    let path = dir.join("replaced.bin");
    fs::write(&path, b"old").expect("a plain file");
    write_file(&path, &schema(Class::Secret), &value).expect("written");
    assert_eq!(mode(&path), 0o600);
    // Nothing is left beside the artifacts.
    assert_eq!(
        listing(&dir),
        [
            "Private.bin",
            "Protocol.bin",
            "Public.bin",
            "Secret.bin",
            "plain",
            "replaced.bin"
        ]
    );
    let _ = fs::remove_dir_all(&dir);
}

/// A value whose encoding writes 1 MiB, past the writer's 64 KiB buffer, so the new file holds
/// part of the envelope, then runs `during` and panics, in the writing pass only.
struct Interrupted<'a> {
    pass: Cell<u32>,
    during: &'a dyn Fn(),
}

impl Encode for Interrupted<'_> {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        let pass = self.pass.get();
        self.pass.set(pass + 1);
        out.put(&vec![0xA5; 1 << 20]);
        if pass == 1 {
            (self.during)();
            panic!("the write is interrupted");
        }
    }
}

#[test]
fn an_interrupted_write_leaves_the_previous_file_whole() {
    let dir = directory("interrupted");
    let path = dir.join("state.twc");
    let secret = schema(Class::Secret);
    let previous = vec![7u8; 100];
    write_file(&path, &secret, &previous).expect("the first file");
    let before = fs::read(&path).expect("the first file");

    let observed = Cell::new(false);
    let during = || {
        // Part way: the target holds the previous file, and a new file beside it, already
        // mode 0600, holds part of the new one.
        assert_eq!(fs::read(&path).expect("the target"), before);
        let names = listing(&dir);
        assert_eq!(names.len(), 2, "{names:?}");
        let temporary = dir.join(
            names
                .iter()
                .find(|n| *n != "state.twc")
                .expect("a new file"),
        );
        // `.state.twc.<16 hex digits>.tmp`: a name no other process can know beforehand.
        let name = temporary
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let digits = name
            .strip_prefix(".state.twc.")
            .and_then(|rest| rest.strip_suffix(".tmp"))
            .expect("the new file's name");
        assert!(
            digits.len() == 16 && digits.bytes().all(|b| b.is_ascii_hexdigit()),
            "{name}"
        );
        assert_eq!(mode(&temporary), 0o600, "never readable by others");
        assert!(fs::metadata(&temporary).expect("metadata").len() >= 1 << 20);
        observed.set(true);
    };
    let value = Interrupted {
        pass: Cell::new(0),
        during: &during,
    };
    let outcome = catch_unwind(AssertUnwindSafe(|| write_file(&path, &secret, &value)));
    assert!(outcome.is_err(), "the encoding panicked");
    assert!(observed.get(), "the write was observed part way");
    // After: the previous file, whole, and nothing else.
    assert_eq!(fs::read(&path).expect("the target"), before);
    assert_eq!(mode(&path), 0o600);
    assert_eq!(listing(&dir), ["state.twc"]);

    // A write that fails at the start, in a directory that does not exist, creates nothing.
    let missing = dir.join("absent").join("state.twc");
    let error = write_file(&missing, &secret, &previous).expect_err("no directory");
    assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    assert_eq!(listing(&dir), ["state.twc"]);
    // A path that names no file is refused.
    let error = write_file(Path::new("/"), &secret, &previous).expect_err("no file name");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);

    // A later write replaces the file whole.
    let next = vec![9u8; 50];
    write_file(&path, &secret, &next).expect("the next file");
    assert_eq!(
        fs::read(&path).expect("the target"),
        envelope::seal(&secret, &next)
    );
    assert_eq!(listing(&dir), ["state.twc"]);
    let _ = fs::remove_dir_all(&dir);
}
