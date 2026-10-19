// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0035's padding zero-check, `0 = Σ_x eq(z, x)·(1 − ψ(x))·w̃(x)`, as a stage's instance: an
//! honest proof of a committed value that is 0 on every padded row passes, whatever it holds on
//! real rows, and a value other than 0 on one padded row, layer or head is rejected at the
//! stage's final check (RFC-0035 'Testing Strategy'). The proofs come from the test kit's
//! reference prover, which multiplies in the public factor `eq(z, x)·(1 − ψ(x))` from its own
//! definition.

use tv_core::selectors::{Rows, padding_zero_check};
use tv_core::sumcheck::{Instance, Prefix, SumcheckError, SumcheckProof, verify_batch};
use tv_field::{Fp, Fp2};
use tv_hash::transcript::{Ascii, Mode, ParamSetId, SessionContext, Transcript};
use tv_testkit::SplitMix64;
use tv_testkit::sumcheck::{self as reference, Fe, Form, Product};

fn context() -> SessionContext<'static> {
    SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: Mode::NonInteractive,
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [3; 32],
        session_id: [4; 16],
    }
}

fn to_fp2(v: Fe) -> Fp2 {
    Fp2::new(
        Fp::from_canonical(v.0).unwrap(),
        Fp::from_canonical(v.1).unwrap(),
    )
}

fn eq_at(r: &[Fe], x: usize) -> Fe {
    r.iter().enumerate().fold(Fe::ONE, |p, (i, &c)| {
        p * if (x >> i) & 1 == 1 { c } else { Fe::ONE - c }
    })
}

/// Rows `[layers, heads, tokens]` of 2 + 1 + 3 variables: 3 of 4 layers, 2 heads, 5 of 8 tokens.
const ROWS: Rows = Rows {
    tokens: (3, 5),
    heads: Some((1, 2)),
    layers: Some((2, 3)),
};

/// Whether entry `x` is a real row: a token below 5 and a layer below 3; both heads are real.
fn real(x: usize) -> bool {
    (x & 7) < 5 && (x >> 4) < 3
}

/// Proves the zero-check of `w` and checks it with an instance whose claim is 0.
fn check(w: &[Fe], seed: u64) -> Result<(), SumcheckError> {
    let mut rng = SplitMix64::new(seed);
    let zero: Vec<Fe> = (0..6)
        .map(|_| Fe::new(rng.next_u64(), rng.next_u64()))
        .collect();
    // eq(z, ·) and 1 − ψ are two multilinear factors: their product, of degree 2 in shared
    // variables, is not one table's extension.
    let eqs: Vec<Fe> = (0..64).map(|x| eq_at(&zero, x)).collect();
    let padding: Vec<Fe> = (0..64)
        .map(|x| if real(x) { Fe::ZERO } else { Fe::ONE })
        .collect();
    let product = Product {
        tables: vec![w.to_vec()],
        public: vec![eqs, padding],
        lambda: None,
        form: Form::Product,
    };
    let mut prover = reference::Transcript::new(&context().to_tce());
    let proof = reference::prove(&mut prover, "stage/1", 3, &[product], None);
    let proof = SumcheckProof {
        rounds: proof
            .rounds
            .iter()
            .map(|r| r.iter().map(|&v| to_fp2(v)).collect())
            .collect(),
        final_evals: proof.evals.iter().map(|&v| to_fp2(v)).collect(),
    };
    let zero: Vec<Fp2> = zero.iter().map(|&v| to_fp2(v)).collect();
    let instance = Instance {
        claim: Fp2::ZERO,
        vars: 6,
        evals: 1,
    };
    let mut transcript = Transcript::new(&context());
    verify_batch(
        &mut transcript,
        Prefix::Stage(1),
        3,
        &[instance],
        &proof,
        |_, point, evals| padding_zero_check(&zero, point, &ROWS, evals[0]),
    )
    .map(|_| ())
}

#[test]
fn a_value_free_on_real_rows_and_zero_on_padded_ones_passes() {
    let mut rng = SplitMix64::new(1);
    let w: Vec<Fe> = (0..64)
        .map(|x| {
            if real(x) {
                Fe::new(rng.next_u64(), rng.next_u64())
            } else {
                Fe::ZERO
            }
        })
        .collect();
    check(&w, 2).unwrap();
}

#[test]
fn a_nonzero_value_on_a_padded_row_layer_or_head_is_rejected() {
    // A padded token of a real layer and head, a padded head, and the padded layer.
    for padded in [5usize, 6 | (1 << 3), 3 << 4, (3 << 4) | 7] {
        assert!(!real(padded));
        let mut w = vec![Fe::ZERO; 64];
        w[padded] = Fe::base(1);
        assert_eq!(
            check(&w, 3),
            Err(SumcheckError::FinalMismatch { round: 6 }),
            "row {padded}"
        );
    }
}

#[test]
fn the_check_refuses_points_that_do_not_fit_its_rows() {
    let point = [Fp2::ONE; 6];
    assert_eq!(
        padding_zero_check(&point, &point[..5], &ROWS, Fp2::ONE),
        None
    );
    assert_eq!(
        padding_zero_check(&point[..5], &point, &ROWS, Fp2::ONE),
        None
    );
    let long = [Fp2::ONE; 7];
    assert_eq!(padding_zero_check(&long, &long, &ROWS, Fp2::ONE), None);
}
