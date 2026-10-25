// SPDX-License-Identifier: Apache-2.0 OR MIT

use tv_core::verdict::ErrorCode;

pub(crate) struct Options<'a> {
    /// Each option with its value, in the order given.
    pairs: Vec<(&'a str, &'a str)>,
    /// The switches given, each once.
    switches: Vec<&'a str>,
}
impl<'a> Options<'a> {
    /// `args` as options of `allowed`, each with one value and given once, and `--json` at most
    /// once; anything else is `ERR-USAGE`.
    pub(crate) fn parse(args: &'a [String], allowed: &[&str]) -> Result<Self, ErrorCode> {
        Self::parse_with(args, allowed, &[], &[])
    }
    /// [`parse`](Self::parse), where the options of `repeated` may be given several times, kept
    /// in their order, and those of `switches` take no value and are given at most once.
    pub(crate) fn parse_with(
        args: &'a [String],
        allowed: &[&str],
        repeated: &[&str],
        switches: &[&str],
    ) -> Result<Self, ErrorCode> {
        let mut pairs = Vec::new();
        let mut on = Vec::new();
        let mut i = 0;
        let mut json = false;
        while let Some(arg) = args.get(i).map(String::as_str) {
            if arg == "--json" {
                if json {
                    return Err(ErrorCode::Usage);
                }
                json = true;
                i += 1;
                continue;
            }
            if switches.contains(&arg) {
                if on.contains(&arg) {
                    return Err(ErrorCode::Usage);
                }
                on.push(arg);
                i += 1;
                continue;
            }
            let once = allowed.contains(&arg);
            if !(once || repeated.contains(&arg))
                || (once && pairs.iter().any(|&(key, _)| key == arg))
            {
                return Err(ErrorCode::Usage);
            }
            let value = args
                .get(i + 1)
                .map(String::as_str)
                .filter(|s| !s.is_empty() && !s.starts_with("--"))
                .ok_or(ErrorCode::Usage)?;
            pairs.push((arg, value));
            i += 2;
        }
        Ok(Self {
            pairs,
            switches: on,
        })
    }
    pub(crate) fn get(&self, key: &str) -> Option<&'a str> {
        self.pairs
            .iter()
            .find_map(|&(k, v)| (k == key).then_some(v))
    }
    /// Every value of a repeated option, in the order given.
    pub(crate) fn all(&self, key: &str) -> Vec<&'a str> {
        self.pairs
            .iter()
            .filter_map(|&(k, v)| (k == key).then_some(v))
            .collect()
    }
    /// Whether the switch `key` was given.
    pub(crate) fn switch(&self, key: &str) -> bool {
        self.switches.contains(&key)
    }
    pub(crate) fn required(&self, key: &str) -> Result<&'a str, ErrorCode> {
        self.get(key).ok_or(ErrorCode::Usage)
    }
    pub(crate) fn number(&self, key: &str, default: u64) -> Result<u64, ErrorCode> {
        self.get(key).map_or(Ok(default), |v| {
            if v.is_empty() || !v.bytes().all(|c| c.is_ascii_digit()) {
                return Err(ErrorCode::Usage);
            }
            v.parse().map_err(|_| ErrorCode::Usage)
        })
    }
    pub(crate) fn required_number(&self, key: &str) -> Result<u64, ErrorCode> {
        self.required(key)?;
        self.number(key, 0)
    }
}

pub(crate) fn digest(text: &str) -> Result<[u8; 32], ErrorCode> {
    hex(text)
}

pub(crate) fn hex<const N: usize>(text: &str) -> Result<[u8; N], ErrorCode> {
    if text.len() != 2 * N {
        return Err(ErrorCode::Usage);
    }
    let mut out = [0; N];
    for (byte, pair) in out.iter_mut().zip(text.as_bytes().chunks_exact(2)) {
        let digit = |x: u8| match x {
            b'0'..=b'9' => Ok(x - b'0'),
            b'a'..=b'f' => Ok(x - b'a' + 10),
            b'A'..=b'F' => Ok(x - b'A' + 10),
            _ => Err(ErrorCode::Usage),
        };
        *byte = digit(pair[0])? * 16 + digit(pair[1])?;
    }
    Ok(out)
}

pub(crate) fn bound(text: &str) -> Result<u64, ErrorCode> {
    let mut parts = text.split('.');
    let integer = parts.next().ok_or(ErrorCode::Usage)?;
    let fraction = parts.next();
    if parts.next().is_some() || integer.is_empty() || !integer.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(ErrorCode::Usage);
    }
    let whole: u64 = integer.parse().map_err(|_| ErrorCode::Usage)?;
    let fractional = if let Some(f) = fraction {
        if f.is_empty() || f.len() > 8 || !f.bytes().all(|b| b.is_ascii_digit()) {
            return Err(ErrorCode::Usage);
        }
        let numerator: u64 = f.parse().map_err(|_| ErrorCode::Usage)?;
        let scaled = numerator * 256;
        let denominator = 10u64.pow(f.len() as u32);
        if !scaled.is_multiple_of(denominator) {
            return Err(ErrorCode::Usage);
        }
        scaled / denominator
    } else {
        0
    };
    whole
        .checked_mul(256)
        .and_then(|n| n.checked_add(fractional))
        .ok_or(ErrorCode::Usage)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A repeated option keeps its values in order, an option of `allowed` is still refused a
    /// second time, and a switch takes no value and is refused a second time.
    #[test]
    fn repeated_options_keep_their_order_and_switches_take_no_value() {
        let args: Vec<String> = ["--b", "z1", "--a", "x", "--s", "--b", "z2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let o = Options::parse_with(&args, &["--a"], &["--b"], &["--s"]).unwrap();
        assert_eq!(o.all("--b"), ["z1", "z2"]);
        assert_eq!(o.get("--a"), Some("x"));
        assert!(o.switch("--s") && !o.switch("--t"));
        assert!(o.all("--a") == ["x"] && o.all("--c").is_empty());
        let none = Options::parse_with(&args[..4], &["--a"], &["--b"], &["--s"]).unwrap();
        assert!(!none.switch("--s"));
        for bad in [
            &["--a", "x", "--a", "y"][..],
            &["--s", "--s"],
            &["--b"],
            &["--b", "--a"],
            &["--s", "value"],
            &["--c", "x"],
        ] {
            let args: Vec<String> = bad.iter().map(|s| s.to_string()).collect();
            assert_eq!(
                Options::parse_with(&args, &["--a"], &["--b"], &["--s"]).err(),
                Some(ErrorCode::Usage),
                "{bad:?}"
            );
        }
        // Without repeats or switches, as before.
        let args: Vec<String> = ["--b", "x"].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            Options::parse(&args, &["--a"]).err(),
            Some(ErrorCode::Usage)
        );
    }
    #[test]
    fn policy_numbers_are_exact_and_cannot_wrap() {
        for (text, value) in [
            ("0", 0),
            ("1.5", 384),
            ("0.00390625", 1),
            ("72057594037927935.99609375", u64::MAX),
        ] {
            assert_eq!(bound(text), Ok(value));
        }
        for bad in [
            "",
            "-1",
            "+1",
            "1.",
            ".5",
            "0.1",
            "1e2",
            "NaN",
            "1.000000000",
            "72057594037927936",
            "999999999999999999999999",
        ] {
            assert_eq!(bound(bad), Err(ErrorCode::Usage), "{bad}");
        }
        assert_eq!(digest(&"A5".repeat(32)), Ok([0xa5; 32]));
        for bad in ["00".repeat(31), "gg".repeat(32), "00".repeat(33)] {
            assert_eq!(digest(&bad), Err(ErrorCode::Usage));
        }
        assert_eq!(hex::<16>(&"Af".repeat(16)), Ok([0xaf; 16]));
        for bad in [
            "00".repeat(15),
            "00".repeat(17),
            "é".repeat(16),
            "gg".repeat(16),
        ] {
            assert_eq!(hex::<16>(&bad), Err(ErrorCode::Usage));
        }
        for (text, value) in [("0", 0), ("18446744073709551615", u64::MAX)] {
            let args = vec!["--n".into(), text.into()];
            assert_eq!(
                Options::parse(&args, &["--n"])
                    .unwrap()
                    .required_number("--n"),
                Ok(value)
            );
        }
        for text in ["-1", "+1", "1.0", "18446744073709551616"] {
            let args = vec!["--n".into(), text.into()];
            assert_eq!(
                Options::parse(&args, &["--n"])
                    .unwrap()
                    .required_number("--n"),
                Err(ErrorCode::Usage)
            );
        }
        assert_eq!(
            Options::parse(&[], &["--n"])
                .unwrap()
                .required_number("--n"),
            Err(ErrorCode::Usage)
        );
    }
}
