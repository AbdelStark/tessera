// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The weight commitment's golden vectors (RFC-0040, 03 §7, `spec/vectors/codec/`): envelopes of
//! `WeightCommitmentPublic` (`0x0301`), each with `R_W`, and of `WeightCommitmentState`
//! (`0x0302`), with their values in the language-neutral JSON of the codec vectors; and states
//! whose `opened_columns` is not strictly increasing, which every reader rejects. The prover's
//! decoder reads both schemas; the verifier's reads `0x0301` only, since it never reads the
//! secret state.
//!
//! The state has no encoding a caller can reach, so the states' bytes are laid out here from
//! RFC-0040's field list ([`state_payload`]); `tessera-pcs`'s own tests check that its encoder
//! seals each case's value, pinned to the case's seed, to the case's bytes.

use serde_json::{Value, json};
use tessera_codec::schema::Schema;
use tessera_codec::{Encode, Sink, encode, envelope, schema};
use tessera_pcs::{OpenedColumns, TypeBudget, WeightCommitmentPublic, WeightLayoutParams};

/// Where the vectors are written.
pub(crate) const PATH: &str = "spec/vectors/codec/weight-commitment.json";

fn number(n: impl ToString) -> Value {
    Value::String(n.to_string())
}

fn bytes(data: &[u8]) -> Value {
    Value::Array(data.iter().map(|&b| number(b)).collect())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// `test-small/1`'s weight layout and `live-64/1`'s (RFC-0026).
const TEST_SMALL: WeightLayoutParams = WeightLayoutParams {
    log2_k: 8,
    log2_inv_rate: 2,
    mask_coeffs: 64,
    blinding_rows: 32,
};
const LIVE_64: WeightLayoutParams = WeightLayoutParams {
    log2_k: 20,
    log2_inv_rate: 2,
    mask_coeffs: 65_536,
    blinding_rows: 4096,
};

/// Ten distinct roots, each byte from `seed`, the root's index and the byte's.
fn roots(seed: u8) -> [[u8; 32]; 10] {
    core::array::from_fn(|t| core::array::from_fn(|i| seed.wrapping_add((13 * t + 7 * i) as u8)))
}

fn publics() -> [(&'static str, WeightCommitmentPublic); 2] {
    [
        (
            "test-small/1, epoch 0",
            WeightCommitmentPublic {
                layout: TEST_SMALL,
                roots: roots(1),
                epoch: 0,
            },
        ),
        (
            "live-64/1, the last epoch",
            WeightCommitmentPublic {
                layout: LIVE_64,
                roots: roots(0xA0),
                epoch: u32::MAX,
            },
        ),
    ]
}

fn columns(list: &[u32]) -> OpenedColumns {
    OpenedColumns::new(list.to_vec()).expect("strictly increasing")
}

/// Budgets with columns opened and rows burned: no column, one, columns 1 and 256, whose
/// little-endian bytes sort the other way, and the ends of `u32`.
fn spent() -> [TypeBudget; 10] {
    let lists: [&[u32]; 10] = [
        &[],
        &[0],
        &[1, 256],
        &[3, 7, 65_535, 65_536],
        &[],
        &[2, 4, 8, 16, 32],
        &[1_000_000],
        &[],
        &[0, u32::MAX],
        &[u32::MAX],
    ];
    let next = [0, 6, 12, 0, 4090, 1, 2, 3, 4089, u32::MAX];
    core::array::from_fn(|t| TypeBudget {
        opened_columns: columns(lists[t]),
        next_blinding_row: next[t],
    })
}

/// One state's fields: its public part, its seed and its budgets.
struct State {
    public: WeightCommitmentPublic,
    seed: [u8; 32],
    budget: [TypeBudget; 10],
}

/// The vectors' states.
fn states() -> [(&'static str, State); 2] {
    let [(_, first), (_, second)] = publics();
    [
        (
            "a new commitment's: no column opened, no row burned",
            State {
                public: first,
                seed: core::array::from_fn(|i| i as u8),
                budget: core::array::from_fn(|_| TypeBudget::default()),
            },
        ),
        (
            "columns opened and blinding rows burned",
            State {
                public: second,
                seed: core::array::from_fn(|i| 0xFF - i as u8),
                budget: spent(),
            },
        ),
    ]
}

/// A state's payload, laid out from RFC-0040's field list rather than by the state's encoder,
/// which no caller reaches: the public part's TCE, the seed's 32 bytes, then the ten budgets,
/// each its opened columns as a `Vec<u32>`, a `u64` count then each index, and its
/// `next_blinding_row` as a `u32`, all little-endian. `columns[t]` is budget `t`'s opened
/// columns, increasing or not. Returns the payload and, if an index is not above the one before
/// it, the first such index's offset counted from the start of the envelope.
fn state_payload(state: &State, columns: [&[u32]; 10]) -> (Vec<u8>, Option<u64>) {
    let mut payload = encode(&state.public);
    payload.extend_from_slice(&state.seed);
    let mut bad = None;
    for (list, budget) in columns.iter().zip(&state.budget) {
        payload.extend_from_slice(&(list.len() as u64).to_le_bytes());
        for (position, column) in list.iter().enumerate() {
            if bad.is_none() && position > 0 && *column <= list[position - 1] {
                bad = Some((envelope::HEADER_LEN + payload.len()) as u64);
            }
            payload.extend_from_slice(&column.to_le_bytes());
        }
        payload.extend_from_slice(&budget.next_blinding_row.to_le_bytes());
    }
    (payload, bad)
}

/// Each budget's opened columns, as the state holds them.
fn opened(state: &State) -> [&[u32]; 10] {
    core::array::from_fn(|t| state.budget[t].opened_columns.columns())
}

/// The state's envelope, sealed around the payload laid out above.
fn state_envelope(state: &State) -> Vec<u8> {
    let (payload, bad) = state_payload(state, opened(state));
    assert_eq!(bad, None, "a state's opened columns increase strictly");
    envelope::seal(&schema::WEIGHT_COMMITMENT_STATE, &Raw(&payload))
}

fn layout_value(layout: &WeightLayoutParams) -> Value {
    json!({
        "log2_k": number(layout.log2_k),
        "log2_inv_rate": number(layout.log2_inv_rate),
        "mask_coeffs": number(layout.mask_coeffs),
        "blinding_rows": number(layout.blinding_rows),
    })
}

fn public_value(public: &WeightCommitmentPublic) -> Value {
    json!({
        "layout": layout_value(&public.layout),
        "roots": public.roots.iter().map(|root| bytes(root)).collect::<Vec<_>>(),
        "epoch": number(public.epoch),
    })
}

/// The state's value.
fn state_value(state: &State) -> Value {
    json!({
        "public": public_value(&state.public),
        "masks_seed": bytes(&state.seed),
        "budget": state.budget.iter().map(|b| json!({
            "opened_columns": b.opened_columns.columns().iter().map(number).collect::<Vec<_>>(),
            "next_blinding_row": number(b.next_blinding_row),
        })).collect::<Vec<_>>(),
    })
}

fn case(name: &str, schema: &Schema, ty: &str, sealed: &[u8], value: Value) -> Value {
    json!({
        "name": name,
        "schema": schema.id.0,
        "version": schema.version,
        "type": ty,
        "hex": hex(sealed),
        "value": value,
    })
}

/// Bytes encoded as they are: a payload the writer cannot produce.
struct Raw<'a>(&'a [u8]);

impl Encode for Raw<'_> {
    fn encode_to<S: Sink + ?Sized>(&self, out: &mut S) {
        out.put(self.0);
    }
}

/// The first state with budget `t`'s opened columns replaced by `list`, which does not increase
/// strictly, sealed, with the offset of the first index not above the one before it.
fn unsorted(t: usize, list: &[u32]) -> (Vec<u8>, u64) {
    let [(_, state), _] = states();
    let mut columns = opened(&state);
    columns[t] = list;
    let (payload, bad) = state_payload(&state, columns);
    let sealed = envelope::seal(&schema::WEIGHT_COMMITMENT_STATE, &Raw(&payload));
    (sealed, bad.expect("an index not above the one before it"))
}

/// `spec/vectors/codec/weight-commitment.json`.
pub(crate) fn render() -> Value {
    let mut cases = Vec::new();
    for (name, public) in publics() {
        let sealed = envelope::seal(&schema::WEIGHT_COMMITMENT_PUBLIC, &public);
        let mut value = case(
            name,
            &schema::WEIGHT_COMMITMENT_PUBLIC,
            "WeightCommitmentPublic",
            &sealed,
            public_value(&public),
        );
        value["weights_root"] = Value::from(hex(&public.weights_root()));
        cases.push(value);
    }
    for (name, state) in states() {
        cases.push(case(
            name,
            &schema::WEIGHT_COMMITMENT_STATE,
            "WeightCommitmentState",
            &state_envelope(&state),
            state_value(&state),
        ));
    }
    let rejected = [
        ("an opened column below the one before it", 0, &[5, 3][..]),
        ("an opened column repeated", 3, &[3, 7, 7, 65_536][..]),
        (
            "columns 256 then 1: sorted by their little-endian bytes, as a map's keys are, but not as integers",
            2,
            &[256, 1][..],
        ),
    ]
    .map(|(why, t, list)| {
        let (sealed, offset) = unsorted(t, list);
        json!({
            "why": why,
            "schema": schema::WEIGHT_COMMITMENT_STATE.id.0,
            "version": schema::WEIGHT_COMMITMENT_STATE.version,
            "type": "WeightCommitmentState",
            "hex": hex(&sealed),
            "code": "ERR-CODEC-NONCANONICAL",
            "offset": offset,
        })
    });
    json!({
        "description": "Weight commitment envelopes (RFC-0040, schemas 0x0301 and 0x0302), written by \
            `cargo xtask gen-vectors` from the Rust types of tessera-pcs. Read under the RFC-0003 \
            message limits, each case's `hex` opens as an envelope of `schema` at `version`, its \
            payload decodes as `type` to `value`, and `value` encodes back to `hex`; for \
            WeightCommitmentPublic, `weights_root` is R_W = H(\"weights-root\", payload). Each \
            `rejected` envelope's payload fails with `code` at `offset`, counted from the start \
            of the envelope. Values follow spec/vectors/codec/README.md 'Values'.",
        "limits": "message",
        "cases": cases,
        "rejected": rejected,
    })
}
