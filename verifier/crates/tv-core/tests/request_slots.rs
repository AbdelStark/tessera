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
