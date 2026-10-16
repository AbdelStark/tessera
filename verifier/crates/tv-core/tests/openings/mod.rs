// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Test support for the opening checks, shared by the `opening_*` integration tests:
//!
//! - the vectors of `spec/vectors/pcs/opening.json`, written by an independent Python oracle,
//!   read into owned openings;
//! - [`Case::check`], which runs the two checks as the verifier's driver will, the evaluation
//!   check and then, on what it returns, the column check;
//! - a reference prover and oracle ([`reference`]), written from RFC-0011 and RFC-0004 for these
//!   tests with the test kit's field and BLAKE3, which compute with no code of `tv-core`,
//!   `tv-field` or `tv-hash`. The prover commits seeded matrices of any shape and opens them
//!   honestly, or commits a matrix whose codewords it changed, as a cheating prover would; the
//!   oracle gives the verdict of any opening.

#![allow(dead_code)]

pub mod reference;

use tv_core::opening::{
    Claim, CombinedRow, Group, Layout, Opening, OpeningError, Params, check_evaluations,
};
use tv_field::{Fp, Fp2};
use tv_hash::Digest;
use tv_testkit::{Json, read_vectors};

// ----- Reading the vectors ----------------------------------------------------------------------

/// The member `key` of a vector object.
pub fn member<'a>(object: &'a Json, key: &str) -> &'a Json {
    object
        .get(key)
        .unwrap_or_else(|| panic!("no member `{key}`"))
}

/// The member `key`, an array.
pub fn list<'a>(object: &'a Json, key: &str) -> &'a [Json] {
    member(object, key)
        .as_array()
        .unwrap_or_else(|| panic!("`{key}` is not an array"))
}

/// The member `key`, a `u64`.
pub fn number(object: &Json, key: &str) -> u64 {
    member(object, key)
        .as_u64()
        .unwrap_or_else(|| panic!("`{key}` is not a u64"))
}

/// The member `key`, a `u32`.
pub fn small(object: &Json, key: &str) -> u32 {
    u32::try_from(number(object, key)).expect("a u32")
}

/// An `Fp` element: `"0x"` and 16 hex digits, canonical.
pub fn fp(text: &Json) -> Fp {
    let word = text.as_hex_u64().expect("an Fp element");
    assert_eq!(
        text.as_str().map(str::len),
        Some(18),
        "an Fp element has 16 digits"
    );
    Fp::from_canonical(word).expect("a canonical Fp element")
}

/// An `Fp2` element `[c0, c1]`.
pub fn fp2(pair: &Json) -> Fp2 {
    match pair.as_array().expect("an Fp2 element") {
        [c0, c1] => Fp2::new(fp(c0), fp(c1)),
        other => panic!("an Fp2 element has two coefficients, not {}", other.len()),
    }
}

/// A list of `Fp2` elements.
pub fn fp2s(values: &Json) -> Vec<Fp2> {
    values.as_array().expect("a list").iter().map(fp2).collect()
}

/// A list of `Fp` elements.
pub fn fps(values: &Json) -> Vec<Fp> {
    values.as_array().expect("a list").iter().map(fp).collect()
}

/// A digest: 64 lowercase hex digits.
pub fn digest(text: &Json) -> Digest {
    let text = text.as_str().expect("a digest");
    assert_eq!(text.len(), 64, "a digest has 64 digits");
    assert!(
        text.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "lowercase hex"
    );
    tv_testkit::hex(text).try_into().expect("32 bytes")
}

/// What the oracle records to locate a divergence.
#[derive(Clone, Debug)]
pub struct Intermediate {
    /// `c_g(row)` for each group, then `ξ_row`, per non-zero row in layout order.
    pub row_coefficients: Vec<Vec<Fp2>>,
    /// `x_j = 7·ω_n^j` per distinct query, increasing.
    pub points: Vec<Fp>,
    /// `leaf_j` per distinct query.
    pub leaves: Vec<Digest>,
    /// `enc(u)(x_j) + Z_H(x_j)·M(x_j)` of each combined row per distinct query.
    pub combined_at_queries: Vec<Vec<Fp2>>,
}

/// One opening: the verifier's inputs and the proof's messages, owned.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: String,
    pub params: Params,
    pub row_vars: u32,
    pub segments: Vec<Vec<u64>>,
    pub root: Digest,
    pub groups: Vec<Group>,
    pub xi_point: Vec<Fp2>,
    pub combined: Vec<CombinedRow>,
    pub queries: Vec<u64>,
    pub columns: Vec<Vec<Fp>>,
    pub siblings: Vec<Digest>,
    pub intermediate: Option<Intermediate>,
}

impl Case {
    pub fn from_json(json: &Json) -> Self {
        let groups = list(json, "groups")
            .iter()
            .map(|group| Group {
                column_point: fp2s(member(group, "column_point")),
                claims: list(group, "claims")
                    .iter()
                    .map(|claim| Claim {
                        row_point: fp2s(member(claim, "row_point")),
                        gamma: fp2(member(claim, "gamma")),
                        value: fp2(member(claim, "value")),
                    })
                    .collect(),
            })
            .collect();
        let intermediate = json.get("intermediate").map(|i| Intermediate {
            row_coefficients: list(i, "row_coefficients").iter().map(fp2s).collect(),
            points: fps(member(i, "points")),
            leaves: list(i, "leaves").iter().map(digest).collect(),
            combined_at_queries: list(i, "combined_at_queries").iter().map(fp2s).collect(),
        });
        Self {
            name: member(json, "name").as_str().expect("a name").to_owned(),
            params: Params {
                log2_k: small(json, "log2_k"),
                log2_inv_rate: small(json, "log2_inv_rate"),
                mask_coeffs: small(json, "mask_coeffs"),
                row_block: small(json, "row_block"),
                commitment_id: digest(member(json, "commitment_id")),
            },
            row_vars: small(json, "row_vars"),
            segments: list(json, "segments")
                .iter()
                .map(|segment| {
                    segment
                        .as_array()
                        .expect("a segment")
                        .iter()
                        .map(|row| row.as_u64().expect("a row"))
                        .collect()
                })
                .collect(),
            root: digest(member(json, "root")),
            groups,
            xi_point: fp2s(member(json, "xi_point")),
            combined: list(json, "combined")
                .iter()
                .map(|row| CombinedRow {
                    u: fp2s(member(row, "u")),
                    mask: fp2s(member(row, "mask")),
                })
                .collect(),
            queries: list(json, "queries")
                .iter()
                .map(|q| q.as_u64().expect("a query"))
                .collect(),
            columns: list(json, "columns").iter().map(fps).collect(),
            siblings: list(json, "siblings").iter().map(digest).collect(),
            intermediate,
        }
    }

    /// The verifier's view of the opening.
    pub fn opening(&self) -> Opening<'_> {
        Opening {
            params: self.params,
            layout: Layout {
                row_vars: self.row_vars,
                segments: &self.segments,
            },
            root: self.root,
            groups: &self.groups,
            xi_point: &self.xi_point,
        }
    }

    /// Both checks, as the driver runs them: the evaluation check, then the column check on
    /// what it returns.
    pub fn check(&self) -> Result<(), OpeningError> {
        check_evaluations(self.opening(), &self.combined)?.check_columns(
            &self.queries,
            &self.columns,
            &self.siblings,
        )
    }

    /// `k`.
    pub fn k(&self) -> usize {
        1 << self.params.log2_k
    }

    /// `n`.
    pub fn n(&self) -> u64 {
        1 << (self.params.log2_k + self.params.log2_inv_rate)
    }

    /// The non-zero rows in layout order.
    pub fn rows(&self) -> Vec<u64> {
        self.segments.iter().flatten().copied().collect()
    }

    /// The sorted, distinct queries.
    pub fn distinct(&self) -> Vec<u64> {
        let mut distinct = self.queries.clone();
        distinct.sort_unstable();
        distinct.dedup();
        distinct
    }

    /// The smallest queried index, as rejections report it.
    pub fn smallest(&self) -> u32 {
        u32::try_from(*self.queries.iter().min().expect("a query")).expect("a u32")
    }
}

/// A changed opening and the verdict the oracle gives it.
#[derive(Clone, Debug)]
pub struct Rejection {
    pub name: String,
    pub case: Case,
    pub expected: OpeningError,
}

/// The vector file.
pub struct Vectors {
    pub json: Json,
    pub openings: Vec<Case>,
    pub rejections: Vec<Rejection>,
}

/// Reads `spec/vectors/pcs/opening.json`.
pub fn vectors() -> Vectors {
    let json = read_vectors("pcs/opening.json");
    let openings = list(&json, "openings")
        .iter()
        .map(Case::from_json)
        .collect();
    let rejections = list(&json, "rejections")
        .iter()
        .map(|rejection| {
            let index = |key: &str| {
                let value = member(rejection, key);
                (!value.is_null()).then(|| value.as_u64().expect("an index or null"))
            };
            let (group, column) = (index("group"), index("column"));
            let reason = member(rejection, "reason").as_str().expect("a reason");
            let expected = match (reason, group, column) {
                ("Evaluation", group, None) => OpeningError::Evaluation {
                    group: group.map(|g| usize::try_from(g).expect("a group")),
                },
                ("MerklePath", None, Some(column)) => OpeningError::MerklePath {
                    column: u32::try_from(column).expect("a column"),
                },
                ("Proximity", None, Some(column)) => OpeningError::Proximity {
                    column: u32::try_from(column).expect("a column"),
                },
                other => panic!("unexpected verdict {other:?}"),
            };
            Rejection {
                name: member(rejection, "name")
                    .as_str()
                    .expect("a name")
                    .to_owned(),
                case: Case::from_json(member(rejection, "opening")),
                expected,
            }
        })
        .collect();
    Vectors {
        json,
        openings,
        rejections,
    }
}
