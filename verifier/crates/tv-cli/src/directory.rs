// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0014 directory exchange: one-shot roles, canonical envelopes, no replacement.
use crate::files::{self, Inputs};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tv_codec::{ENVELOPE_VERSION, Expected, MAGIC};
use tv_core::verdict::ErrorCode;
use tv_hash::{Domain, hash};

pub(crate) const PACKETS: [&str; 7] = [
    "00-SeedCommit.tce",
    "01-Round1.tce",
    "01-ProverSeedShare.tce",
    "02-SeedReveal.tce",
    "03-Round2Response.tce",
    "04-ColumnQueries.tce",
    "05-Round3Response.tce",
];

pub(crate) struct Directory {
    path: PathBuf,
    pub(crate) inputs: Inputs,
    timeout: Option<Duration>,
}
impl Directory {
    pub(crate) fn new(
        path: &Path,
        inputs: Inputs,
        timeout: Option<Duration>,
    ) -> Result<Self, ErrorCode> {
        if !fs::metadata(path).map_err(|_| ErrorCode::Io)?.is_dir() {
            return Err(ErrorCode::Io);
        }
        if PACKETS
            .iter()
            .any(|name| fs::symlink_metadata(path.join(name)).is_ok())
        {
            return Err(ErrorCode::Usage);
        }
        files::publish(&path.join(".tessera-verifier"), b"")?;
        Ok(Self {
            path: path.to_owned(),
            inputs,
            timeout,
        })
    }
    pub(crate) fn send(
        &self,
        index: usize,
        schema: Expected,
        payload: &[u8],
    ) -> Result<(), ErrorCode> {
        files::publish(&self.path.join(PACKETS[index]), &seal(schema, payload)?)
    }
    pub(crate) fn receive(&mut self, index: usize) -> Result<Vec<u8>, ErrorCode> {
        let start = Instant::now();
        let limit =
            self.timeout
                .unwrap_or(Duration::from_secs(if index == 4 { 3600 } else { 600 }));
        let path = self.path.join(PACKETS[index]);
        loop {
            if start.elapsed() >= limit {
                return Err(ErrorCode::ProtocolTimeout);
            }
            match fs::symlink_metadata(&path) {
                Ok(_) => return self.inputs.read(&path),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(_) => return Err(ErrorCode::Io),
            }
        }
    }
}

pub(crate) fn seal(schema: Expected, payload: &[u8]) -> Result<Vec<u8>, ErrorCode> {
    let len = payload.len().checked_add(52).ok_or(ErrorCode::Memory)?;
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| ErrorCode::Memory)?;
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&ENVELOPE_VERSION.to_le_bytes());
    out.extend_from_slice(&schema.schema_id.to_le_bytes());
    out.extend_from_slice(&schema.version.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&hash(Domain::Envelope, &out));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tv_testkit::hex;
    #[test]
    fn canonical_messages_match_the_independent_protocol_oracle() {
        let vectors = tv_testkit::read_vectors("protocol/protocol.json");
        for m in vectors.get("messages").unwrap().as_array().unwrap() {
            let expected = hex(m.get("envelope").unwrap().as_str().unwrap());
            let payload = hex(m.get("payload").unwrap().as_str().unwrap());
            let id = u16::from_le_bytes([expected[6], expected[7]]);
            assert_eq!(
                seal(
                    Expected {
                        schema_id: id,
                        version: 1
                    },
                    &payload
                )
                .unwrap(),
                expected
            );
        }
    }
    #[test]
    fn reused_roles_packets_and_zero_deadlines_fail_closed() {
        let path =
            std::env::temp_dir().join(format!("tessera-tv-directory-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        let inputs = || Inputs {
            remaining: 1024,
            max_artifact: 1024,
        };
        let mut d = Directory::new(&path, inputs(), Some(Duration::ZERO)).unwrap();
        fs::write(path.join(PACKETS[1]), b"arrived").unwrap();
        assert_eq!(d.receive(1), Err(ErrorCode::ProtocolTimeout));
        assert!(matches!(
            Directory::new(&path, inputs(), None),
            Err(ErrorCode::Usage)
        ));
        d.send(0, tv_core::protocol::SeedCommit::SCHEMA, &[3; 48])
            .unwrap();
        let before = fs::read(path.join(PACKETS[0])).unwrap();
        assert_eq!(
            d.send(0, tv_core::protocol::SeedCommit::SCHEMA, &[4; 48]),
            Err(ErrorCode::Usage)
        );
        assert_eq!(fs::read(path.join(PACKETS[0])).unwrap(), before);
        fs::remove_file(path.join(".tessera-verifier")).unwrap();
        assert!(matches!(
            Directory::new(&path, inputs(), None),
            Err(ErrorCode::Usage)
        ));
        assert_eq!(fs::read_dir(&path).unwrap().count(), 2);
        fs::remove_dir_all(path).unwrap();
    }
}
