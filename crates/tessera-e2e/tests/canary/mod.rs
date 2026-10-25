// SPDX-License-Identifier: Apache-2.0 OR MIT

//! OBS-3 (05 §3): the canary scan. The canary model (`spec/vectors/canary/`) carries runs of eight
//! weights whose bytes form a distinctive 64-bit value, and its recorded tokens distinctive ids.
//! The scan looks for each canary in decimal, hex and little-endian byte forms.
//!
//! A run's forms are long enough that chance never spells them: its bytes, the weights widened
//! to 16, 32 and 64 bits or as field elements, and its value in decimal or hex, anywhere in an
//! artifact; and, in text, its weights as consecutive integers. A token id has 12 bits, and its
//! short forms are spelled by chance. In text, a token id counts as a whole word in decimal or
//! hex, since a random digest holds short runs of digits. In binary artifacts, a lone id cannot
//! be told from a public integer such as a column index, so the scan looks for each recorded
//! sequence of ids instead, as `u32` words and as field elements.
use serde_json::Value;

/// Goldilocks, which a weight enters as a field element.
const P: i128 = 0xFFFF_FFFF_0000_0001;

enum Form {
    /// These bytes, anywhere.
    Bytes(Vec<u8>),
    /// A word, between characters other than ASCII letters and digits, that spells this number.
    Decimal(String),
    /// A word that spells this number in hex, in either case, a `0x` and leading zeros aside.
    Hex(String),
    /// These integers, consecutive among the integers a text spells.
    Integers(Vec<i128>),
}

pub struct Canaries(Vec<(String, Form)>);

impl Canaries {
    /// The canaries `canaries.json` lists: the model's weight runs, and the ids of the recorded
    /// and of the altered token files.
    pub fn new(manifest: &Value) -> Self {
        let mut forms = Vec::new();
        for run in manifest["weight_runs"].as_array().unwrap() {
            let weights: Vec<i8> = run["weights"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| i8::try_from(w.as_i64().unwrap()).unwrap())
                .collect();
            let bytes: Vec<u8> = weights.iter().map(|&w| w as u8).collect();
            let value = u64::from_le_bytes(bytes.as_slice().try_into().unwrap());
            assert_eq!(format!("{value:016x}"), run["value"].as_str().unwrap());
            let name = format!(
                "{} {} weights {value:016x}",
                run["tensor"].as_str().unwrap(),
                run["layer"]
            );
            let widened = |width: usize| -> Vec<u8> {
                weights
                    .iter()
                    .flat_map(|&w| i64::from(w).to_le_bytes()[..width].to_vec())
                    .collect()
            };
            let field: Vec<u8> = weights
                .iter()
                .flat_map(|&w| (i128::from(w).rem_euclid(P) as u64).to_le_bytes())
                .collect();
            let hex = |b: &[u8]| -> String { b.iter().map(|x| format!("{x:02x}")).collect() };
            let mut add = |form: &str, f: Form| forms.push((format!("{name} {form}"), f));
            add("little-endian bytes", Form::Bytes(bytes.clone()));
            for width in [2, 4, 8] {
                add(
                    &format!("as {}-bit integers", 8 * width),
                    Form::Bytes(widened(width)),
                );
            }
            add("as field elements", Form::Bytes(field));
            add("decimal", Form::Bytes(value.to_string().into_bytes()));
            add(
                "signed decimal",
                Form::Bytes((value as i64).to_string().into_bytes()),
            );
            for text in [format!("{value:016x}"), hex(&bytes)] {
                add("hex", Form::Bytes(text.clone().into_bytes()));
                add("hex", Form::Bytes(text.to_uppercase().into_bytes()));
            }
            add(
                "as integers",
                Form::Integers(weights.iter().map(|&w| w.into()).collect()),
            );
        }
        let recorded = ids(&manifest["token_ids"]);
        let altered = ids(&manifest["mismatch"]["token_ids"]);
        let prompt = usize::try_from(manifest["prompt_len"].as_u64().unwrap()).unwrap();
        let mut tokens: Vec<u32> = recorded.iter().chain(&altered).copied().collect();
        tokens.sort_unstable();
        tokens.dedup();
        for t in tokens {
            forms.push((format!("token {t} decimal"), Form::Decimal(t.to_string())));
            forms.push((format!("token {t} hex"), Form::Hex(format!("{t:x}"))));
        }
        for (name, sequence) in [
            ("prompt", &recorded[..prompt]),
            ("continuation", &recorded[prompt..]),
            ("altered continuation", &altered[prompt..]),
        ] {
            forms.push((
                format!("{name} {sequence:?} as u32 words"),
                Form::Bytes(sequence.iter().flat_map(|t| t.to_le_bytes()).collect()),
            ));
            forms.push((
                format!("{name} {sequence:?} as field elements"),
                Form::Bytes(
                    sequence
                        .iter()
                        .flat_map(|&t| u64::from(t).to_le_bytes())
                        .collect(),
                ),
            ));
        }
        Self(forms)
    }

    /// The canary forms found in `bytes`; text forms count only when `bytes` is UTF-8.
    pub fn find(&self, bytes: &[u8]) -> Vec<&str> {
        let text = std::str::from_utf8(bytes).is_ok();
        let integers = if text { integers(bytes) } else { Vec::new() };
        let words = bytes
            .split(|b| !b.is_ascii_alphanumeric())
            .filter(|w| text && !w.is_empty());
        let found = |form: &Form| match form {
            Form::Bytes(b) => bytes.windows(b.len()).any(|w| w == b.as_slice()),
            Form::Decimal(d) => words.clone().any(|w| w == d.as_bytes()),
            Form::Hex(h) => words.clone().any(|w| {
                let w = match w {
                    [b'0', b'x' | b'X', rest @ ..] => rest,
                    _ => w,
                };
                let start = w.iter().position(|&c| c != b'0').unwrap_or(w.len());
                w[start..].eq_ignore_ascii_case(h.as_bytes())
            }),
            Form::Integers(v) => integers.windows(v.len()).any(|w| w == v.as_slice()),
        };
        self.0
            .iter()
            .filter(|(_, form)| found(form))
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

fn ids(list: &Value) -> Vec<u32> {
    list.as_array()
        .unwrap()
        .iter()
        .map(|t| u32::try_from(t.as_u64().unwrap()).unwrap())
        .collect()
}

/// The integers a text spells, a minus sign right before the digits included; a number too long
/// for an `i128` is left out.
fn integers(bytes: &[u8]) -> Vec<i128> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let digits = std::str::from_utf8(&bytes[start..i]).unwrap();
            if let Ok(n) = digits.parse::<i128>() {
                out.push(if start > 0 && bytes[start - 1] == b'-' {
                    -n
                } else {
                    n
                });
            }
        } else {
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canaries() -> Canaries {
        let manifest =
            std::fs::read(super::super::root().join("spec/vectors/canary/canaries.json"));
        Canaries::new(&serde_json::from_slice(&manifest.unwrap()).unwrap())
    }

    /// The scan finds a leak injected in each form, in text or in binary bytes as the form
    /// allows, and nothing in the artifact without it.
    #[test]
    fn the_scan_finds_each_injected_canary() {
        let c = canaries();
        let line = br#"{"event":"tessera.prover.stage_proved","fields":{"id":3,"rounds":8},"ts":"2026-10-25T19:02:25.832140Z"}"#;
        let mut binary: Vec<u8> = (0..4096u32)
            .map(|i| (i.wrapping_mul(0x9E37) >> 7) as u8)
            .collect();
        binary[17] = 0xFF;
        assert!(std::str::from_utf8(&binary).is_err());
        assert!(c.find(line).is_empty());
        assert!(c.find(&binary).is_empty());
        let texts: [&[u8]; 11] = [
            b"weights 12926923227794424666",
            b"w=-5519820845915126950 ",
            b"0xb365a82f976ec35a",
            b"dump 5AC36E972FA865B3 end",
            b"[90, -61, 110, -105, 47, -88, 101, -77]",
            b"\"token\":2211,",
            b"token=4085",
            b"id 0x0ff5",
            b"t:C07;",
            b"next a3d",
            b"[2620]",
        ];
        for leak in texts {
            let mut text = line[..line.len() - 1].to_vec();
            text.extend_from_slice(b",\"note\":\"");
            text.extend_from_slice(leak);
            text.extend_from_slice(b"\"}");
            assert_eq!(c.find(&text).len(), 1, "{}", String::from_utf8_lossy(leak));
        }
        let words = |ts: &[u32]| -> Vec<u8> { ts.iter().flat_map(|t| t.to_le_bytes()).collect() };
        let field = |ts: &[u64]| -> Vec<u8> { ts.iter().flat_map(|t| t.to_le_bytes()).collect() };
        let leaks: [Vec<u8>; 9] = [
            vec![0x5A, 0xC3, 0x6E, 0x97, 0x2F, 0xA8, 0x65, 0xB3],
            [-99i16, 83, -52, 117, -41, 66, -114, 29]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
            [117i32, 109, 103, -98, -111, 95, -106, -121]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
            [90i64, -61, 110, -105, 47, -88, 101, -77]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
            [90i64, -61, 110, -105, 47, -88, 101, -77]
                .iter()
                .flat_map(|&w| (i128::from(w).rem_euclid(P) as u64).to_le_bytes())
                .collect(),
            field(&[2211, 4085, 3079]),
            words(&[2211, 4085, 3079]),
            words(&[2621, 2621, 2621]),
            words(&[2621, 2621, 2620]),
        ];
        for leak in &leaks {
            let mut bytes = binary.clone();
            bytes.splice(1000..1000, leak.iter().copied());
            assert_eq!(c.find(&bytes).len(), 1, "{leak:02x?}");
        }
        // A lone id in binary bytes, and a word that only contains an id's digits, are chance.
        let mut bytes = binary.clone();
        bytes.splice(1000..1000, *b"2211 c07 ");
        bytes.splice(2000..2000, 2211u32.to_le_bytes());
        bytes.splice(3000..3000, 2211u64.to_le_bytes());
        assert!(c.find(&bytes).is_empty());
        assert!(
            c.find(b"22110 12211 0x8a30 8a3f 30790 622db2211c 9e4c07 x8a3")
                .is_empty()
        );
    }
}
