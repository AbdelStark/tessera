// SPDX-License-Identifier: Apache-2.0 OR MIT

//! RFC-0071 selectors and request windows against the independent slot oracle.
mod claim_support;
use claim_support::{items, list, member, number, text};
use tv_core::gadgets::greedy::{Window, selected_active_mle, window_mle};
use tv_core::scoring::window_uhat_sum;
use tv_core::selectors::{causal_slots, lt, slot_selector};
use tv_field::{Fp, Fp2};
use tv_testkit::{Json, read_vectors};

fn field(value: &Json) -> Fp2 {
    let [a, b] = items(value) else {
        panic!("Fp2 pair")
    };
    Fp2::new(Fp::reduce(number(a)), Fp::reduce(number(b)))
}
fn point(value: &Json) -> Vec<Fp2> {
    items(value).iter().map(field).collect()
}

#[test]
fn public_selectors_match_independent_request_slot_vectors() {
    let vectors = read_vectors("request-slots/slots.json");
    let cases = list(&vectors, "predicates");
    assert_eq!(cases.len(), 15);
    for case in cases {
        let name = text(case, "name");
        let bits = number(member(case, "slot_bits"));
        let lengths: Vec<u32> = list(case, "lengths").iter().map(number).collect();
        let token = point(member(case, "r_token"));
        let key = point(member(case, "r_key"));
        let head = point(member(case, "r_head"));
        let heads = number(member(case, "heads"));
        let rows = slot_selector(&token, bits, &lengths).expect("slot geometry");
        let causal = causal_slots(&token, &key, bits, &lengths).expect("causal geometry");
        let head_factor = lt(&head, heads).expect("head cube");
        for (actual, key) in [
            (rows, "slot_selector"),
            (rows * head_factor, "head_row_selector"),
            (causal, "causal_selector"),
            (causal * head_factor, "head_causal_selector"),
            (Fp2::ONE - rows * head_factor, "softmax_root"),
        ] {
            assert_eq!(actual, field(member(case, key)), "{name}: {key}");
        }
    }
}

#[test]
fn selected_windows_and_scoring_match_independent_vectors() {
    let vectors = read_vectors("request-slots/slots.json");
    let cases = list(&vectors, "greedy");
    assert_eq!(cases.len(), 9);
    for case in cases {
        let name = text(case, "name");
        let bits = number(member(case, "slot_bits"));
        let time = point(member(case, "r_time"));
        let windows: Vec<_> = list(case, "requests")
            .iter()
            .enumerate()
            .map(|(r, req)| Window {
                ordinal: u32::try_from(r).expect("ordinal"),
                prompt: number(member(req, "prompt_len")),
                outputs: number(member(req, "output_len")),
            })
            .collect();
        let active: Vec<_> = windows.iter().copied().filter(|w| w.outputs != 0).collect();
        assert_eq!(
            selected_active_mle(&active, bits, &time),
            Some(field(member(case, "selected_active_mle"))),
            "{name}"
        );
        for column in list(case, "key_columns") {
            let window = Window {
                ordinal: number(member(column, "request_index")),
                prompt: number(member(column, "prompt_len")),
                outputs: number(member(column, "output_len")),
            };
            assert_eq!(
                window_mle(window, bits, &time),
                Some(field(member(column, "window_mle"))),
                "{name}"
            );
        }
        let bounds: Vec<u64> = list(case, "row_bounds").iter().map(number).collect();
        let expected = list(case, "window_u_hat");
        assert_eq!(windows.len(), expected.len());
        for (window, sum) in windows.iter().zip(expected) {
            assert_eq!(
                window_uhat_sum(
                    &bounds,
                    number(member(case, "slot_len")),
                    window.ordinal,
                    window.prompt,
                    window.outputs
                )
                .map(|v| v.0),
                Some(number(sum)),
                "{name}"
            );
        }
    }
}

#[test]
fn slot_selectors_refuse_invalid_public_geometry() {
    assert_eq!(slot_selector(&[], 0, &[]), None);
    assert_eq!(slot_selector(&[], 1, &[1]), None);
    assert_eq!(slot_selector(&[], 0, &[1, 1]), None);
    assert_eq!(slot_selector(&[Fp2::ZERO], 1, &[3]), None);
    assert_eq!(causal_slots(&[Fp2::ZERO], &[], 1, &[1]), None);
    assert_eq!(
        window_mle(
            Window {
                ordinal: 0,
                prompt: 0,
                outputs: 1
            },
            1,
            &[Fp2::ZERO]
        ),
        None
    );
}

#[test]
fn request_window_refuses_an_ordinal_outside_the_request_cube() {
    // One request bit represents ordinals 0 and 1 only; ordinal 2 must not alias 0.
    assert_eq!(
        window_mle(
            Window {
                ordinal: 2,
                prompt: 1,
                outputs: 1
            },
            1,
            &[Fp2::ZERO, Fp2::ZERO]
        ),
        None,
    );
}

#[cfg(feature = "test-params")]
#[test]
fn allocation_free_batch_count_matches_runtime_layout_for_oracle_and_seeded_batches() {
    use tv_core::{gadgets::runtime, lowering, protocol::RequestRef, validation};
    use tv_testkit::SplitMix64;
    let vectors = read_vectors("request-slots/slots.json");
    for model in ["tiny-2-minimal.d", "tiny-4-servable.d"] {
        let claims = claim_support::claims_of(model);
        let validated = validation::validate(&claims).unwrap();
        let mut cases: Vec<Vec<u32>> = list(&vectors, "lowering")
            .iter()
            .filter(|case| {
                text(case, "model") == model && number::<u16>(member(case, "protocol")) == 1
            })
            .map(|case| list(case, "lengths").iter().map(number).collect())
            .collect();
        let mut rng = SplitMix64::new(0x551);
        for count in 1..=3 {
            for _ in 0..12 {
                cases.push(
                    (0..count)
                        .map(|_| {
                            1 + (rng.next_u64() % u64::from(claims.arch.max_context.min(16))) as u32
                        })
                        .collect(),
                );
            }
        }
        for lengths in cases {
            for outputless in [false, true] {
                let requests: Vec<_> = lengths
                    .iter()
                    .enumerate()
                    .map(|(r, &length)| RequestRef {
                        request_id: r as u64,
                        session_id: [0; 16],
                        prompt_len: if outputless { length } else { 1 },
                        output_len: if outputless { 0 } else { length - 1 },
                    })
                    .collect();
                let count = validated.batch_advice_rows(&requests).unwrap();
                let graph =
                    lowering::lower_batch(&validation::check_public(&claims).unwrap(), &lengths)
                        .unwrap();
                let pairs: Vec<_> = requests
                    .iter()
                    .map(|r| (r.prompt_len, r.output_len))
                    .collect();
                let runtime = runtime::Plan::derive_batch(
                    &graph,
                    claims.gadget_policy,
                    &pairs,
                    validated.tables(),
                )
                .unwrap();
                let layout = runtime
                    .advice_layout(tv_core::advice::Config::new(
                        &claims.arch,
                        claims.gadget_policy,
                        claims.layout.advice.log2_k,
                        claims.layout.advice.segments,
                    ))
                    .unwrap();
                assert_eq!(
                    count,
                    layout.rows(),
                    "{model}, lengths={lengths:?}, outputless={outputless}"
                );
            }
        }
    }
}
