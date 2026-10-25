// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Batching (RFC-0016 'Batching'): first-fit decreasing on total tokens within each group, so
//! that both parties can reproduce a sample's batches.

use super::memory;
use super::sample::RequestKey;
use crate::anchor::{Failure, usage};

/// A selected request to batch: its key, its group and its total tokens `L`.
///
/// The greedy requests form one group, and each pinned sampler key's temperature requests
/// another (RFC-0061 'Which pin applies'); `G` names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BatchRequest<G> {
    /// The request.
    pub key: RequestKey,
    /// Its group.
    pub group: G,
    /// Its total tokens over both directions and both periods of its window.
    pub tokens: u64,
}

/// One batch, one proof: its group, its requests in `RequestKey` order, as its statement lists
/// them, and their total tokens.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Batch<G> {
    /// The group of every request of the batch.
    pub group: G,
    /// The requests, in `RequestKey` order.
    pub requests: Vec<RequestKey>,
    /// Their total tokens.
    pub tokens: u64,
}

/// An opened batch while a group is packed. A batch opened by a request longer than
/// `max_batch_tokens` is already past it, so no other request fits in it.
struct Open {
    tokens: u64,
    requests: Vec<RequestKey>,
}

/// Packs `requests` as RFC-0016 'Batching' fixes, group by group in increasing order of `G`:
///
/// - within a group, requests go in decreasing `L`, ties in `RequestKey` order;
/// - each goes into the first batch, in the order the batches were opened, whose total stays at
///   most `max_batch_tokens` with it, else into a new batch;
/// - a request longer than `max_batch_tokens` is a batch on its own, which takes no other.
///
/// The batches of a group follow in the order they were opened, each listing its requests in
/// `RequestKey` order. Under a policy's bounds, `max_batch_tokens ≥ 2 · min_batch_tokens`, at
/// most one batch of each group has fewer than `min_batch_tokens` tokens: two such batches would
/// total under `max_batch_tokens`, so the later one's first request would have fit in the
/// earlier one.
///
/// # Errors
///
/// `Inconclusive(ERR-USAGE)` for a request listed twice, and `Inconclusive(ERR-RES-MEMORY)` when
/// memory cannot be reserved.
pub fn pack<G: Ord + Copy>(
    max_batch_tokens: u32,
    requests: &[BatchRequest<G>],
) -> Result<Vec<Batch<G>>, Failure> {
    let max = u64::from(max_batch_tokens);
    let mut order: Vec<&BatchRequest<G>> = Vec::new();
    order
        .try_reserve_exact(requests.len())
        .map_err(|_| memory())?;
    order.extend(requests.iter());
    let mut keys: Vec<RequestKey> = Vec::new();
    keys.try_reserve_exact(requests.len())
        .map_err(|_| memory())?;
    keys.extend(requests.iter().map(|r| r.key));
    keys.sort_unstable();
    if keys.windows(2).any(|w| matches!(w, [a, b] if a == b)) {
        return Err(usage());
    }
    // Group, then decreasing tokens, then the key.
    order.sort_unstable_by(|a, b| {
        a.group
            .cmp(&b.group)
            .then(b.tokens.cmp(&a.tokens))
            .then(a.key.cmp(&b.key))
    });
    let mut out: Vec<Batch<G>> = Vec::new();
    for group in order.chunk_by(|a, b| a.group == b.group) {
        let Some(first) = group.first() else {
            continue;
        };
        let mut opened: Vec<Open> = Vec::new();
        for request in group {
            // A request longer than the maximum fits in no batch, and opens its own.
            let target = opened.iter_mut().find(|b| {
                b.tokens
                    .checked_add(request.tokens)
                    .is_some_and(|total| total <= max)
            });
            match target {
                Some(batch) => {
                    batch.tokens = batch.tokens.saturating_add(request.tokens);
                    batch.requests.try_reserve(1).map_err(|_| memory())?;
                    batch.requests.push(request.key);
                }
                None => {
                    opened.try_reserve(1).map_err(|_| memory())?;
                    let mut keys = Vec::new();
                    keys.try_reserve(1).map_err(|_| memory())?;
                    keys.push(request.key);
                    opened.push(Open {
                        tokens: request.tokens,
                        requests: keys,
                    });
                }
            }
        }
        out.try_reserve(opened.len()).map_err(|_| memory())?;
        for mut batch in opened {
            batch.requests.sort_unstable();
            out.push(Batch {
                group: first.group,
                requests: batch.requests,
                tokens: batch.tokens,
            });
        }
    }
    Ok(out)
}
