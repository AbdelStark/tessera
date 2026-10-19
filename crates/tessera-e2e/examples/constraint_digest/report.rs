// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Location-aware differences between public diagnostic trees.
use serde_json::Value;
use std::collections::BTreeSet;

/// The earliest differing semantic entry, in sorted-field / array-position order.
#[derive(Debug, PartialEq, Eq)]
pub struct Difference {
    pub path: String,
    pub compiler: Option<Value>,
    pub verifier: Option<Value>,
}

/// Compare public metadata, retaining the first differing location and both values.
pub fn difference(compiler: &Value, verifier: &Value) -> Option<Difference> {
    fn at(path: &str, a: &Value, b: &Value) -> Option<Difference> {
        if a == b {
            return None;
        }
        match (a, b) {
            (Value::Array(xs), Value::Array(ys)) => {
                for i in 0..xs.len().max(ys.len()) {
                    let p = format!("{path}/{i}");
                    match (xs.get(i), ys.get(i)) {
                        (Some(x), Some(y)) => {
                            if let Some(d) = at(&p, x, y) {
                                return Some(d);
                            }
                        }
                        (x, y) => {
                            return Some(Difference {
                                path: p,
                                compiler: x.cloned(),
                                verifier: y.cloned(),
                            });
                        }
                    }
                }
            }
            (Value::Object(xs), Value::Object(ys)) => {
                let keys: BTreeSet<_> = xs.keys().chain(ys.keys()).collect();
                for key in keys {
                    let p = format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"));
                    match (xs.get(key), ys.get(key)) {
                        (Some(x), Some(y)) => {
                            if let Some(d) = at(&p, x, y) {
                                return Some(d);
                            }
                        }
                        (x, y) => {
                            return Some(Difference {
                                path: p,
                                compiler: x.cloned(),
                                verifier: y.cloned(),
                            });
                        }
                    }
                }
            }
            _ => {
                return Some(Difference {
                    path: path.to_owned(),
                    compiler: Some(a.clone()),
                    verifier: Some(b.clone()),
                });
            }
        }
        None
    }
    at("", compiler, verifier)
}
