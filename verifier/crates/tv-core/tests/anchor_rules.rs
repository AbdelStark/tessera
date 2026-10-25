// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A batch's anchor checks, each rule broken alone, with the request and kind it names:
//! RFC-0062's session rules, RFC-0063's request rules across a period boundary, RFC-0059's frame
//! checks and the statement's lengths, and RFC-0061's sampler pins, at load and for containment;
//! and the cases those RFCs' Testing Strategy sections say must not be refused.

mod anchor_support;

use anchor_support::{
    GENESIS_NS, Item, PERIOD_NS, Scenario, carry, close, end, inbound, no_pins, open, outbound, req,
};
use tv_core::anchor::{
    Failure, PinError, PinRule, PinnedSamplerKey, SamplerPins, SamplerPolicy, check_batch,
};
use tv_core::protocol::{Decoding, PredictorSpec};
use tv_core::verdict::{AnchorFailure, ErrorCode, RejectReason, VerifierError};

use AnchorFailure::{Frames, SamplerKey, Session};

fn rejected(request_id: u64, kind: AnchorFailure) -> Result<(), Failure> {
    Err(Failure::Rejected(RejectReason::Anchor {
        request_id: Some(request_id),
        kind,
    }))
}

fn usage() -> Failure {
    Failure::Inconclusive(VerifierError {
        code: ErrorCode::Usage,
    })
}

/// Request 7 of session 1: 3 tokens in, 2 + 1 out, then its end.
fn honest(s: u8, r: u64) -> Vec<Item> {
    vec![
        inbound(s, r, 0, 3).item(),
        outbound(s, r, 0, 2).item(),
        outbound(s, r, 1, 1).item(),
        end(s, r),
    ]
}

fn with_open(s: u8, rest: Vec<Item>) -> Vec<Item> {
    [vec![open(s)], rest].concat()
}

#[test]
fn an_honest_request_passes() {
    let one = Scenario::one([with_open(1, honest(1, 7)), vec![close(1)]].concat());
    assert_eq!(one.greedy(vec![req(1, 7, 3, 3)]), Ok(()));
    // An empty batch has nothing to check.
    assert_eq!(one.greedy(vec![]), Ok(()));
    // Two sessions interleaved, each with its requests.
    let two = Scenario::one(vec![
        open(1),
        open(2),
        inbound(2, 1, 0, 4).item(),
        inbound(1, 7, 0, 3).item(),
        outbound(2, 1, 0, 0).item(),
        outbound(1, 7, 0, 2).item(),
        end(2, 1),
        outbound(1, 7, 1, 1).item(),
        end(1, 7),
    ]);
    assert_eq!(two.greedy(vec![req(1, 7, 3, 3), req(2, 1, 4, 0)]), Ok(()));
}

// ----- RFC-0062: the session rules --------------------------------------------------------------

/// RFC-0062 'Testing Strategy': a session with no open or carry, with two opens, with an open and
/// a carry, with its open after a frame; each `Session`.
#[test]
fn each_key_rule_break_is_session() {
    let cases: [(&str, Vec<Item>); 6] = [
        ("no open or carry", honest(1, 7)),
        ("two opens", [vec![open(1), open(1)], honest(1, 7)].concat()),
        (
            "an open and a carry",
            [vec![carry(1), open(1)], honest(1, 7)].concat(),
        ),
        (
            "two carries",
            [vec![carry(1), carry(1)], honest(1, 7)].concat(),
        ),
        (
            "its open after a frame",
            [
                vec![inbound(1, 7, 0, 3).item(), open(1)],
                honest(1, 7)[1..].to_vec(),
            ]
            .concat(),
        ),
        (
            "its open after its close",
            [vec![close(1), open(1)], honest(1, 7)].concat(),
        ),
    ];
    for (name, items) in cases {
        let one = Scenario::one(items);
        assert_eq!(
            one.greedy(vec![req(1, 7, 3, 3)]),
            rejected(7, Session),
            "{name}"
        );
    }
    // A second open after the session's frames is still a second key.
    let one = Scenario::one([with_open(1, honest(1, 7)), vec![open(1)]].concat());
    assert_eq!(one.greedy(vec![req(1, 7, 3, 3)]), rejected(7, Session));
}

/// RFC-0062 rules 2 and 3, `Frames`, over all the session's frames in the period: a duplicate
/// position or counter of a request outside the batch is the session's, and names the batch's
/// first request in the session.
#[test]
fn duplicate_positions_and_counters_are_frames() {
    // Two frames at one position of the request itself, with different counters.
    let items = with_open(1, [vec![inbound(1, 7, 0, 3).item()], honest(1, 7)].concat());
    assert_eq!(
        Scenario::one(items).greedy(vec![req(1, 7, 6, 3)]),
        rejected(7, Frames)
    );
    // At one position of request 9, not in the batch: the batch's requests of the session.
    let items = with_open(
        1,
        [
            honest(1, 5),
            vec![inbound(1, 9, 0, 1).item(), inbound(1, 9, 0, 2).item()],
            honest(1, 7),
        ]
        .concat(),
    );
    let one = Scenario::one(items);
    assert_eq!(
        one.greedy(vec![req(1, 7, 3, 3), req(1, 5, 3, 3)]),
        rejected(7, Frames)
    );
    assert_eq!(
        one.greedy(vec![req(1, 5, 3, 3), req(1, 7, 3, 3)]),
        rejected(5, Frames)
    );
    // Two frames with one counter, within the request.
    let items = vec![
        open(1),
        inbound(1, 7, 0, 3).counter(4),
        outbound(1, 7, 0, 2).counter(4),
        outbound(1, 7, 1, 1).item(),
        end(1, 7),
    ];
    assert_eq!(
        Scenario::one(items).greedy(vec![req(1, 7, 3, 3)]),
        rejected(7, Frames)
    );
    // Across two requests of the batch: the first of the session in batch order is named.
    let items = vec![
        open(1),
        inbound(1, 7, 0, 3).counter(4),
        inbound(1, 8, 0, 1).counter(4),
        outbound(1, 7, 0, 2).item(),
        outbound(1, 8, 0, 1).item(),
        outbound(1, 7, 1, 1).item(),
        end(1, 7),
        end(1, 8),
    ];
    let one = Scenario::one(items);
    assert_eq!(
        one.greedy(vec![req(1, 7, 3, 3), req(1, 8, 1, 1)]),
        rejected(7, Frames)
    );
    assert_eq!(
        one.greedy(vec![req(1, 8, 1, 1), req(1, 7, 3, 3)]),
        rejected(8, Frames)
    );
    // Another session's request before them in the batch passes and is not named.
    let items = [
        vec![open(2)],
        honest(2, 1),
        vec![
            open(1),
            inbound(1, 7, 0, 3).counter(4),
            inbound(1, 8, 0, 1).counter(4),
            outbound(1, 7, 0, 2).item(),
            outbound(1, 8, 0, 1).item(),
            outbound(1, 7, 1, 1).item(),
            end(1, 7),
            end(1, 8),
        ],
    ]
    .concat();
    let one = Scenario::one(items);
    assert_eq!(
        one.greedy(vec![req(2, 1, 3, 3), req(1, 8, 1, 1), req(1, 7, 3, 3)]),
        rejected(8, Frames)
    );
}

/// RFC-0062 'Not refused': a frame logged after its session's close in the same period, and a
/// carried session's frame whose counter is below a frame's of the previous period.
#[test]
fn frames_after_a_close_and_lower_counters_after_a_carry_pass() {
    let items = vec![
        open(1),
        inbound(1, 7, 0, 3).item(),
        outbound(1, 7, 0, 2).item(),
        close(1),
        outbound(1, 7, 1, 1).item(),
        end(1, 7),
    ];
    assert_eq!(Scenario::one(items).greedy(vec![req(1, 7, 3, 3)]), Ok(()));
    let two = Scenario::two(
        vec![
            open(1),
            inbound(1, 7, 0, 1).counter(10),
            outbound(1, 7, 0, 1).counter(11),
            end(1, 7),
        ],
        vec![
            carry(1),
            inbound(1, 8, 0, 1).counter(5),
            outbound(1, 8, 0, 1).counter(6),
            end(1, 8),
        ],
    );
    assert_eq!(two.greedy(vec![req(1, 8, 1, 1)]), Ok(()));
}

// ----- RFC-0063: the request's rules ------------------------------------------------------------

/// RFC-0063 'Testing Strategy', 'The rules': each break alone, with the kind it names.
#[test]
fn each_request_rule_break_is_refused() {
    let one: [(&str, Vec<Item>, u32, u32, AnchorFailure); 10] = [
        ("no end", honest(1, 7)[..3].to_vec(), 3, 3, Frames),
        (
            "two ends",
            [honest(1, 7), vec![end(1, 7)]].concat(),
            3,
            3,
            Frames,
        ),
        (
            "a frame after the end",
            [honest(1, 7), vec![outbound(1, 7, 2, 0).item()]].concat(),
            3,
            3,
            Frames,
        ),
        (
            "an end before every frame",
            [vec![end(1, 7)], honest(1, 7)[..3].to_vec()].concat(),
            3,
            3,
            Frames,
        ),
        (
            "a late inbound frame",
            vec![
                inbound(1, 7, 0, 3).item(),
                outbound(1, 7, 0, 2).item(),
                inbound(1, 7, 1, 0).item(),
                outbound(1, 7, 1, 1).item(),
                end(1, 7),
            ],
            3,
            3,
            Frames,
        ),
        (
            "a gap in the indices",
            vec![
                inbound(1, 7, 0, 3).item(),
                outbound(1, 7, 0, 2).item(),
                outbound(1, 7, 2, 1).item(),
                end(1, 7),
            ],
            3,
            3,
            Frames,
        ),
        (
            "indices not from 0",
            vec![
                inbound(1, 7, 1, 3).item(),
                outbound(1, 7, 0, 2).item(),
                outbound(1, 7, 1, 1).item(),
                end(1, 7),
            ],
            3,
            3,
            Frames,
        ),
        ("a prompt length one short", honest(1, 7), 2, 3, Frames),
        ("an output length one long", honest(1, 7), 3, 4, Frames),
        ("no frames at all", vec![end(1, 7)], 0, 0, Frames),
    ];
    for (name, items, prompt, output, kind) in one {
        let scenario = Scenario::one(with_open(1, items));
        assert_eq!(
            scenario.greedy(vec![req(1, 7, prompt, output)]),
            rejected(7, kind),
            "{name}"
        );
    }
    // RFC-0059: a length no plaintext has, 22 + 4n for no n, whatever the statement's lengths.
    for length in [0, 21, 23, 24, 25] {
        let items = with_open(
            1,
            vec![
                inbound(1, 7, 0, 3).length(length),
                outbound(1, 7, 0, 2).item(),
                end(1, 7),
            ],
        );
        let scenario = Scenario::one(items);
        for prompt in [0, 3, (length.saturating_sub(22)) / 4] {
            assert_eq!(
                scenario.greedy(vec![req(1, 7, prompt, 2)]),
                rejected(7, Frames),
                "{length}"
            );
        }
    }
}

/// RFC-0063 'The rules' across the boundary: an end in `P − 1`, two key commitments across it,
/// two opens in `P − 1`, one position logged in both periods, and a counter used in both.
#[test]
fn each_window_rule_break_is_refused() {
    // An end in P − 1: the request ended there, and P holds no end of it.
    let two = Scenario::two(with_open(1, honest(1, 7)), vec![carry(1)]);
    assert_eq!(two.greedy(vec![req(1, 7, 3, 3)]), rejected(7, Frames));
    // An end in each period.
    let two = Scenario::two(
        vec![open(1), inbound(1, 7, 0, 3).item(), end(1, 7)],
        vec![carry(1), outbound(1, 7, 0, 3).item(), end(1, 7)],
    );
    assert_eq!(two.greedy(vec![req(1, 7, 3, 3)]), rejected(7, Frames));
    // Two key commitments across the boundary.
    let two = Scenario::two(
        vec![open(1), inbound(1, 7, 0, 3).item()],
        vec![
            Item::Carry {
                s: 1,
                key: [0xEE; 32],
            },
            outbound(1, 7, 0, 3).item(),
            end(1, 7),
        ],
    );
    assert_eq!(two.greedy(vec![req(1, 7, 3, 3)]), rejected(7, Session));
    // Two opens in P − 1.
    let two = Scenario::two(
        vec![open(1), open(1), inbound(1, 7, 0, 3).item()],
        vec![carry(1), outbound(1, 7, 0, 3).item(), end(1, 7)],
    );
    assert_eq!(two.greedy(vec![req(1, 7, 3, 3)]), rejected(7, Session));
    // No carry in P, where the request has a frame.
    let two = Scenario::two(
        vec![open(1), inbound(1, 7, 0, 3).item()],
        vec![outbound(1, 7, 0, 3).item(), end(1, 7)],
    );
    assert_eq!(two.greedy(vec![req(1, 7, 3, 3)]), rejected(7, Session));
    // One position logged in both periods: each period alone has it once.
    let two = Scenario::two(
        vec![open(1), inbound(1, 7, 0, 3).item()],
        vec![
            carry(1),
            inbound(1, 7, 0, 3).item(),
            outbound(1, 7, 0, 3).item(),
            end(1, 7),
        ],
    );
    assert_eq!(two.greedy(vec![req(1, 7, 3, 3)]), rejected(7, Frames));
    assert_eq!(two.greedy(vec![req(1, 7, 6, 3)]), rejected(7, Frames));
    // A counter used in both periods by the batch's frames of the session (RFC-0059's distinct
    // counters of a session's frames in the batch), named at the session's first request.
    let two = Scenario::two(
        vec![open(1), inbound(1, 7, 0, 3).counter(4)],
        vec![
            carry(1),
            outbound(1, 7, 0, 3).counter(4),
            end(1, 7),
            inbound(1, 5, 0, 1).counter(9),
            outbound(1, 5, 0, 1).counter(10),
            end(1, 5),
        ],
    );
    assert_eq!(two.greedy(vec![req(1, 7, 3, 3)]), rejected(7, Frames));
    assert_eq!(
        two.greedy(vec![req(1, 5, 1, 1), req(1, 7, 3, 3)]),
        rejected(5, Frames)
    );
    // The same counter in P − 1 for a request outside the batch is not the batch's.
    let two = Scenario::two(
        vec![open(1), inbound(1, 3, 0, 1).counter(4), end(1, 3)],
        vec![
            carry(1),
            inbound(1, 7, 0, 3).counter(4),
            outbound(1, 7, 0, 3).item(),
            end(1, 7),
        ],
    );
    assert_eq!(two.greedy(vec![req(1, 7, 3, 3)]), Ok(()));
}

/// RFC-0063 'The window': a request whose frames lie in `P − 1` and `P`, its end in `P`, is
/// accepted; its part in `P − 1`, read alone, is not a candidate of `P − 1`. An end before the
/// request's frames is refused by rule 2, but is still a candidate.
#[test]
fn a_request_across_the_boundary_is_one_request() {
    let previous = vec![
        open(1),
        inbound(1, 7, 0, 3).item(),
        outbound(1, 7, 0, 2).item(),
    ];
    let current = vec![carry(1), outbound(1, 7, 1, 1).item(), end(1, 7)];
    let two = Scenario::two(previous.clone(), current);
    assert_eq!(two.greedy(vec![req(1, 7, 3, 3)]), Ok(()));
    // The part in P − 1 alone, as a statement would have counted it, is short.
    assert_eq!(two.greedy(vec![req(1, 7, 3, 2)]), rejected(7, Frames));
    let alone = Scenario::one(previous);
    assert_eq!(alone.window().candidates().unwrap(), vec![]);
    assert_eq!(alone.greedy(vec![req(1, 7, 3, 2)]), rejected(7, Frames));
    // An end logged before the request's frames: a candidate, refused by rule 2.
    let early = Scenario::one(with_open(
        1,
        [vec![end(1, 7)], honest(1, 7)[..3].to_vec()].concat(),
    ));
    let candidates = early.window().candidates().unwrap();
    assert_eq!(candidates.len(), 1);
    assert!(candidates[0].ended);
    assert_eq!(candidates[0].tokens, 6);
    assert_eq!(early.greedy(vec![req(1, 7, 3, 3)]), rejected(7, Frames));
    // A request in P − 1 with no end in either period is P's unterminated candidate, and is
    // refused by rule 1.
    let open_ended = Scenario::two(vec![open(1), inbound(1, 7, 0, 3).item()], vec![carry(1)]);
    let candidates = open_ended.window().candidates().unwrap();
    assert_eq!(candidates.len(), 1);
    assert!(!candidates[0].ended);
    assert_eq!(
        open_ended.greedy(vec![req(1, 7, 3, 0)]),
        rejected(7, Frames)
    );
}

/// The order: a request's session first, then its frames, then its sampler key; requests in batch
/// order; a session's break named by the first request of the batch that reads it.
#[test]
fn the_first_failure_in_order_is_the_verdict() {
    // A session break and a frame break on one request: Session.
    let one = Scenario::one([vec![open(1), open(1)], honest(1, 7)[..3].to_vec()].concat());
    assert_eq!(one.greedy(vec![req(1, 7, 0, 0)]), rejected(7, Session));
    // A good request, then a bad one, then a worse one: the bad one.
    let items = [
        with_open(1, honest(1, 5)),
        with_open(2, honest(2, 7)[..3].to_vec()),
        honest(3, 9),
    ]
    .concat();
    let one = Scenario::one(items);
    assert_eq!(
        one.greedy(vec![req(1, 5, 3, 3), req(2, 7, 3, 3), req(3, 9, 3, 3)]),
        rejected(7, Frames)
    );
    assert_eq!(
        one.greedy(vec![req(1, 5, 3, 3), req(3, 9, 3, 3), req(2, 7, 3, 3)]),
        rejected(9, Session)
    );
    // A session that breaks a rule in P names the batch's first request of the session.
    let one = Scenario::one([vec![open(1), open(1)], honest(1, 7), honest(1, 8)].concat());
    assert_eq!(
        one.greedy(vec![req(1, 8, 3, 3), req(1, 7, 3, 3)]),
        rejected(8, Session)
    );
    // A break in P − 1 only is read by the requests with a frame there.
    let two = Scenario::two(
        vec![open(1), open(1), inbound(1, 7, 0, 3).item()],
        vec![
            carry(1),
            outbound(1, 7, 0, 3).item(),
            end(1, 7),
            inbound(1, 5, 0, 1).item(),
            outbound(1, 5, 0, 1).item(),
            end(1, 5),
        ],
    );
    assert_eq!(
        two.greedy(vec![req(1, 5, 1, 1), req(1, 7, 3, 3)]),
        rejected(7, Session)
    );
    // Frames before the sampler key: a request with a gap and outside every pin.
    let one = Scenario::one(with_open(1, honest(1, 7)[..3].to_vec()));
    assert_eq!(
        one.sampled(vec![req(1, 7, 3, 3)], &no_pins()),
        rejected(7, Frames)
    );
}

#[test]
fn a_statement_anchored_elsewhere_is_the_verifiers_input() {
    let one = Scenario::one(with_open(1, honest(1, 7)));
    let mut statement = one.statement(vec![req(1, 7, 3, 3)], anchor_support::greedy());
    statement.manifest_digest[0] ^= 1;
    assert_eq!(
        check_batch(&statement, &one.window(), &no_pins()),
        Err(usage())
    );
}

// ----- RFC-0061: the sampler pins ---------------------------------------------------------------

/// `P` runs `[6_000_000, 7_000_000)`.
const P_START: u64 = GENESIS_NS + PERIOD_NS;

fn pin(from_ns: u64, window_ns: u64, grace_ns: u64) -> PinnedSamplerKey {
    PinnedSamplerKey {
        commitment: [0xC0; 32],
        class_bits: 0,
        from_ns,
        window_ns,
        grace_ns,
        received_ns: 0,
        nonce: [0x0E; 32],
    }
}

fn policy() -> SamplerPolicy {
    SamplerPolicy::default_for(PERIOD_NS)
}

/// Two adjacent windows of 400 µs and a third after a gap, each with 1 µs of grace.
fn three_pins() -> SamplerPins {
    SamplerPins::load(
        vec![
            pin(P_START, 400_000, 1_000),
            pin(P_START + 400_000, 400_000, 1_000),
            pin(P_START + 900_000, 50_000, 1_000),
        ],
        policy(),
    )
    .unwrap()
}

/// Request `r` of session 1 with frames at `times`, inbound first, then its end.
fn timed(r: u64, times: &[u64]) -> Vec<Item> {
    let mut items = Vec::new();
    for (i, &t) in times.iter().enumerate() {
        let frame = if i == 0 {
            inbound(1, r, 0, 1)
        } else {
            outbound(1, r, i as u32 - 1, 1)
        };
        items.push(frame.at(t));
    }
    items.push(Item::End {
        s: 1,
        r,
        t: times.last().map(|t| t + 1),
    });
    items
}

fn lengths(r: u64, times: &[u64]) -> tv_core::protocol::RequestRef {
    req(1, r, 1, times.len() as u32 - 1)
}

/// RFC-0061 'Testing Strategy' for the pins: each `SamplerKey` case alone, and the edges.
#[test]
fn each_request_outside_its_pin_is_sampler_key() {
    let pins = three_pins();
    let check = |times: &[u64]| {
        let scenario = Scenario::one(with_open(1, timed(7, times)));
        scenario.sampled(vec![lengths(7, times)], &pins)
    };
    let a = P_START;
    // Inside the first window.
    assert_eq!(check(&[a + 100, a + 200, a + 300]), Ok(()));
    // The edges: the earliest frame at from_ns, a frame at from_ns + W + G − 1.
    assert_eq!(check(&[a, a + 400_000 + 1_000 - 1]), Ok(()));
    // A later frame at from_ns + W + G.
    assert_eq!(
        check(&[a + 10, a + 400_000 + 1_000]),
        rejected(7, SamplerKey)
    );
    // Before the first pin.
    assert_eq!(check(&[a - 1, a + 10]), rejected(7, SamplerKey));
    // In the gap between the second and the third windows.
    assert_eq!(check(&[a + 850_000, a + 850_010]), rejected(7, SamplerKey));
    // After the last window's end.
    assert_eq!(check(&[a + 950_000, a + 950_010]), rejected(7, SamplerKey));
    // The earliest frame need not be inbound: the earliest of all, In and Out alike.
    let scenario = Scenario::one(with_open(
        1,
        vec![
            inbound(1, 7, 0, 1).at(a + 400_100),
            outbound(1, 7, 0, 1).at(a + 399_000),
            end(1, 7),
        ],
    ));
    // seq order puts the inbound frame first; times are the recorder's and need not follow it.
    assert_eq!(scenario.sampled(vec![req(1, 7, 1, 1)], &pins), Ok(()));
    // A greedy statement needs no pin.
    let scenario = Scenario::one(with_open(1, timed(7, &[a - 1, a + 10])));
    assert_eq!(scenario.greedy(vec![lengths(7, &[a - 1, a + 10])]), Ok(()));
}

#[test]
fn a_batch_shares_its_first_requests_pin() {
    let pins = three_pins();
    let a = P_START;
    let items = [
        with_open(1, timed(7, &[a + 100, a + 200])),
        timed(8, &[a + 300, a + 310]),
        timed(9, &[a + 400_100, a + 400_200]),
    ]
    .concat();
    let scenario = Scenario::one(items);
    let (r7, r8, r9) = (
        lengths(7, &[0, 0]),
        lengths(8, &[0, 0]),
        lengths(9, &[0, 0]),
    );
    assert_eq!(
        scenario.sampled(vec![r7.clone(), r8.clone()], &pins),
        Ok(())
    );
    assert_eq!(
        scenario.sampled(vec![r7.clone(), r8.clone(), r9.clone()], &pins),
        rejected(9, SamplerKey)
    );
    assert_eq!(
        scenario.sampled(vec![r9.clone(), r7.clone()], &pins),
        rejected(7, SamplerKey)
    );
    // The mixture predictor decodes with temperature too.
    let mixture = PredictorSpec::Mixture {
        decoding: Decoding::Temperature {
            t: tv_codec::Rational::new(1, 2).unwrap(),
        },
        eps_log2: 8,
    };
    let statement = scenario.statement(vec![r7.clone(), r9.clone()], mixture);
    assert_eq!(
        check_batch(&statement, &scenario.window(), &pins),
        rejected(9, SamplerKey)
    );
    let statement = scenario.statement(
        vec![r7.clone(), r9.clone()],
        PredictorSpec::Mixture {
            decoding: Decoding::Greedy,
            eps_log2: 8,
        },
    );
    assert_eq!(check_batch(&statement, &scenario.window(), &pins), Ok(()));
}

/// A request whose frames lie in `P − 1` and `P`, its pin the window of its earliest frame there.
#[test]
fn a_request_across_the_boundary_takes_its_earliest_frames_pin() {
    let before = GENESIS_NS + PERIOD_NS - 300;
    let pins = SamplerPins::load(
        vec![pin(GENESIS_NS + PERIOD_NS / 2, PERIOD_NS, 1_000)],
        policy(),
    )
    .unwrap();
    let two = Scenario::two(
        vec![open(1), inbound(1, 7, 0, 1).at(before)],
        vec![carry(1), outbound(1, 7, 0, 1).at(P_START + 10), end(1, 7)],
    );
    assert_eq!(two.sampled(vec![req(1, 7, 1, 1)], &pins), Ok(()));
    // With the pin starting after the frame in P − 1, the request falls within none.
    let late = SamplerPins::load(vec![pin(P_START, PERIOD_NS, 1_000)], policy()).unwrap();
    assert_eq!(
        two.sampled(vec![req(1, 7, 1, 1)], &late),
        rejected(7, SamplerKey)
    );
}

/// RFC-0061 'The policy applies to every verifier': a pin whose `b` or `G` exceeds the policy's
/// is absent; it still loads, so a change of policy invalidates no pin held, and an adopted list
/// holding such a pin loads too.
#[test]
fn a_pin_outside_the_policy_is_absent() {
    let a = P_START;
    let scenario = Scenario::one(with_open(1, timed(7, &[a + 100, a + 200])));
    let request = vec![lengths(7, &[0, 0])];
    let mut classed = pin(a, 400_000, 1_000);
    classed.class_bits = 1;
    let pins = SamplerPins::load(vec![classed], policy()).unwrap();
    assert_eq!(
        scenario.sampled(request.clone(), &pins),
        rejected(7, SamplerKey)
    );
    let lenient = SamplerPolicy {
        max_class_bits: 8,
        ..policy()
    };
    let pins = SamplerPins::load(vec![classed], lenient).unwrap();
    assert_eq!(scenario.sampled(request.clone(), &pins), Ok(()));
    // A grace above the policy's: present under a larger max_grace_ns, absent under a smaller.
    let graced = pin(a, 400_000, 2_000);
    let strict = SamplerPolicy {
        max_grace_ns: 1_999,
        ..policy()
    };
    let pins = SamplerPins::load(vec![graced], strict).unwrap();
    assert_eq!(
        scenario.sampled(request.clone(), &pins),
        rejected(7, SamplerKey)
    );
    let pins = SamplerPins::load(vec![graced], policy()).unwrap();
    assert_eq!(scenario.sampled(request, &pins), Ok(()));
    // The window's length is not compared with the policy's after arrival.
    let other_length = SamplerPolicy {
        window_ns: 10_000_000,
        max_grace_ns: 1_000,
        ..policy()
    };
    SamplerPins::load(vec![pin(a, 400_000, 1_000)], other_length).unwrap();
}

/// A change to a pin list.
type PinEdit = dyn Fn(&mut Vec<PinnedSamplerKey>);

/// RFC-0061 'Checked at load': each rule broken alone is `Inconclusive(ERR-USAGE)`.
#[test]
fn each_load_rule_refuses_alone() {
    let a = P_START;
    let good = vec![pin(a, 400_000, 1_000), pin(a + 400_000, 400_000, 1_000)];
    SamplerPins::load(good.clone(), policy()).unwrap();
    SamplerPins::load(Vec::new(), policy()).unwrap();
    let broken = |mutate: &PinEdit| {
        let mut pins = good.clone();
        mutate(&mut pins);
        SamplerPins::load(pins, policy()).unwrap_err()
    };
    let at = |index, rule| PinError::Pin { index, rule };
    let cases: [(&str, &PinEdit, PinError); 10] = [
        (
            "a retroactive pin",
            &|p| p[1].received_ns = p[1].from_ns + 1,
            at(1, PinRule::Retroactive),
        ),
        (
            "a pin inside the previous window",
            &|p| p[1].from_ns -= 1,
            at(1, PinRule::Overlap),
        ),
        (
            "a pin before the previous one",
            &|p| p.swap(0, 1),
            at(1, PinRule::Overlap),
        ),
        ("W = 0", &|p| p[0].window_ns = 0, at(0, PinRule::Window)),
        (
            "2·G > W",
            &|p| p[1].grace_ns = 200_001,
            at(1, PinRule::Grace),
        ),
        (
            "a G whose double overflows",
            &|p| p[0].grace_ns = u64::MAX,
            at(0, PinRule::Grace),
        ),
        (
            "b = 65",
            &|p| p[0].class_bits = 65,
            at(0, PinRule::ClassBits),
        ),
        (
            "an overflowing from_ns + W + G",
            &|p| {
                p.truncate(1);
                p[0].from_ns = u64::MAX - 400_000;
                p[0].received_ns = 0;
            },
            at(0, PinRule::Overflow),
        ),
        (
            "W below twice the previous pin's G",
            &|p| {
                p[0].grace_ns = 150_000;
                p[1].window_ns = 299_999;
                p[1].grace_ns = 0;
            },
            at(1, PinRule::ShortWindow),
        ),
        (
            "the first rule broken is reported",
            &|p| {
                p[1].window_ns = 0;
                p[1].from_ns -= 1;
            },
            at(1, PinRule::Window),
        ),
    ];
    for (name, mutate, expected) in cases {
        let error = broken(mutate);
        assert_eq!(error, expected, "{name}");
        assert_eq!(error.failure(), usage(), "{name}");
        assert_eq!(Failure::from(error), usage());
    }
    // A pin may start exactly at the previous window's end, and at its own receipt.
    let mut edge = good.clone();
    edge[1].received_ns = edge[1].from_ns;
    SamplerPins::load(edge, policy()).unwrap();
    // 2·G = W, and W exactly twice the previous G.
    let edge = vec![pin(a, 400_000, 200_000), pin(a + 400_000, 400_000, 0)];
    SamplerPins::load(
        edge,
        SamplerPolicy {
            max_grace_ns: 200_000,
            ..policy()
        },
    )
    .unwrap();

    // The policy's rules.
    let refused = |p: SamplerPolicy| SamplerPins::load(good.clone(), p).unwrap_err();
    assert_eq!(
        refused(SamplerPolicy {
            window_ns: 0,
            ..policy()
        }),
        PinError::PolicyWindow
    );
    assert_eq!(
        refused(SamplerPolicy {
            max_grace_ns: policy().window_ns / 2 + 1,
            ..policy()
        }),
        PinError::PolicyGrace
    );
    assert_eq!(
        refused(SamplerPolicy {
            max_grace_ns: u64::MAX,
            ..policy()
        }),
        PinError::PolicyGrace
    );
    assert_eq!(
        refused(SamplerPolicy {
            max_class_bits: 65,
            ..policy()
        }),
        PinError::PolicyClassBits
    );
    SamplerPins::load(
        good.clone(),
        SamplerPolicy {
            max_class_bits: 64,
            ..policy()
        },
    )
    .unwrap();
    // The policy before the pins: both broken, the policy is reported.
    let mut bad = good.clone();
    bad[0].window_ns = 0;
    assert_eq!(
        SamplerPins::load(
            bad,
            SamplerPolicy {
                window_ns: 0,
                ..policy()
            }
        )
        .unwrap_err(),
        PinError::PolicyWindow
    );
}

#[test]
fn the_default_policy_follows_the_period() {
    let day = 86_400_000_000_000;
    assert_eq!(
        SamplerPolicy::default_for(day),
        SamplerPolicy {
            window_ns: day,
            max_grace_ns: 600_000_000_000,
            max_class_bits: 0
        }
    );
    assert_eq!(
        SamplerPolicy::default_for(1_000),
        SamplerPolicy {
            window_ns: 1_000,
            max_grace_ns: 500,
            max_class_bits: 0
        }
    );
}
