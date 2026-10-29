// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Batching (RFC-0016 'Batching', RFC-0071 'The batching'): first-fit decreasing per slot class
//! within each group, so that both parties can reproduce a sample's batches.

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

/// Request `r`'s slot class `c_r = ⌈log2 max(L_r, 1)⌉` (`0..=64`, RFC-0071 'The batching').
fn slot_class(tokens: u64) -> u32 {
    u64::BITS.saturating_sub(tokens.saturating_sub(1).leading_zeros())
}

/// Maximum requests of `class` whose counted tokens `2^{class + ⌈log2 R⌉}` stay at most
/// `max_batch_tokens` (`1` when `max_batch_tokens == 0` or `class > ⌊log2 max_batch_tokens⌋`,
/// RFC-0071 'The batching').
fn class_capacity(max_batch_tokens: u32, class: u32) -> usize {
    if max_batch_tokens == 0 {
        return 1;
    }
    let floor_log = u32::BITS
        .saturating_sub(1)
        .saturating_sub(max_batch_tokens.leading_zeros());
    let Some(shift) = floor_log.checked_sub(class) else {
        return 1;
    };
    1u64.checked_shl(shift)
        .and_then(|cap| usize::try_from(cap).ok())
        .unwrap_or(usize::MAX)
        .max(1)
}

/// Packs `requests` as RFC-0016 'Batching' and RFC-0071 'The batching' fix, group by group in
/// increasing order of `G` and slot class by slot class in decreasing order of
/// `c_r = ⌈log2 max(L_r, 1)⌉`:
///
/// - within a `(group, c)` class, requests go in decreasing `L`, ties in `RequestKey` order;
/// - a batch of `R` requests of class `c` counts `2^{c + ⌈log2 R⌉}` tokens against
///   `max_batch_tokens`;
/// - each goes into the first batch of its class whose counted tokens stay at most
///   `max_batch_tokens` with it, else into a new batch;
/// - a request of class `c > ⌊log2 max_batch_tokens⌋` is a batch on its own.
///
/// Because counted tokens within class `c` depend only on `c` and the request count `R`, every
/// batch in class `c` has uniform capacity `2^{⌊log2 max_batch_tokens⌋ − c}` (or `1` when
/// `c > ⌊log2 max_batch_tokens⌋`), so first-fit decreasing fills batches sequentially in
/// `O(n log n)` total work. Each batch lists its requests in `RequestKey` order and records their
/// saturating sum of real `tokens`.
///
/// # Errors
///
/// `Inconclusive(ERR-USAGE)` for a request listed twice, and `Inconclusive(ERR-RES-MEMORY)` when
/// memory cannot be reserved.
pub fn pack<G: Ord + Copy>(
    max_batch_tokens: u32,
    requests: &[BatchRequest<G>],
) -> Result<Vec<Batch<G>>, Failure> {
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
    // Group, then decreasing slot class, then decreasing tokens, then the key.
    order.sort_unstable_by(|a, b| {
        a.group
            .cmp(&b.group)
            .then_with(|| slot_class(b.tokens).cmp(&slot_class(a.tokens)))
            .then_with(|| b.tokens.cmp(&a.tokens))
            .then_with(|| a.key.cmp(&b.key))
    });
    let mut out: Vec<Batch<G>> = Vec::new();
    for class_slice in
        order.chunk_by(|a, b| a.group == b.group && slot_class(a.tokens) == slot_class(b.tokens))
    {
        let Some(first) = class_slice.first() else {
            continue;
        };
        let cap = class_capacity(max_batch_tokens, slot_class(first.tokens));
        let batch_count = class_slice.len().div_ceil(cap);
        out.try_reserve(batch_count).map_err(|_| memory())?;
        for chunk in class_slice.chunks(cap) {
            let mut batch_keys = Vec::new();
            batch_keys
                .try_reserve_exact(chunk.len())
                .map_err(|_| memory())?;
            let mut tokens = 0u64;
            for req in chunk {
                tokens = tokens.saturating_add(req.tokens);
                batch_keys.push(req.key);
            }
            batch_keys.sort_unstable();
            out.push(Batch {
                group: first.group,
                requests: batch_keys,
                tokens,
            });
        }
    }
    Ok(out)
}
