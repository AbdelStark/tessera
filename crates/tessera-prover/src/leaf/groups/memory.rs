// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public construction bounds for RFC-0054 grouping metadata (#677).

use core::mem::{align_of, size_of};

use tessera_protocol::LeafGroups;
use tessera_sumcheck::MergePolynomial;
use tessera_tables::TableId;

use super::{Block, GroupRead, GroupedLayout, InstanceLayout, LeafError, LeafLayout, Share};

/// Checked requested bytes for a public grouping or one-group layout (RFC-0018/0054).
/// Borrowed inputs, inline objects, payloads and allocator/runtime overhead are additional.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupLayoutMemory {
    peak: u64,
    output: u64,
}

impl GroupLayoutMemory {
    /// Peak requested bytes, including retained output and construction overlap.
    pub fn peak_bytes(self) -> u64 {
        self.peak
    }

    /// Requested bytes retained by the returned layout, including spare capacity.
    pub fn output_bytes(self) -> u64 {
        self.output
    }
}

fn add(a: u64, b: u64) -> Result<u64, LeafError> {
    a.checked_add(b).ok_or(LeafError::Memory)
}

fn mul(a: u64, b: u64) -> Result<u64, LeafError> {
    a.checked_mul(b).ok_or(LeafError::Memory)
}

fn array<T>(len: u64) -> Result<u64, LeafError> {
    let bytes = mul(len, size_of::<T>() as u64)?;
    if usize::try_from(len).is_err() || bytes > isize::MAX as u64 {
        return Err(LeafError::Memory);
    }
    Ok(bytes)
}

fn capacity<T>(len: u64) -> Result<u64, LeafError> {
    if len == 0 || size_of::<T>() == 0 {
        return Ok(0);
    }
    // Rust 1.95 RawVec's minimum nonzero capacity and doubling growth.
    let minimum = match size_of::<T>() {
        1 => 8,
        2..=1024 => 4,
        _ => 1,
    };
    len.max(minimum)
        .checked_next_power_of_two()
        .ok_or(LeafError::Memory)
}

fn pushed<T>(len: u64) -> Result<u64, LeafError> {
    array::<T>(capacity::<T>(len)?)
}

fn old_capacity<T>(len: u64) -> Result<u64, LeafError> {
    let final_cap = capacity::<T>(len)?;
    let initial_cap = capacity::<T>(1)?;
    array::<T>(if final_cap > initial_cap {
        final_cap / 2
    } else {
        0
    })
}

impl GroupedLayout {
    /// Requested peak heap bytes of [`Self::new`], including its returned metadata.
    ///
    /// Prices public stage shapes only, without allocation, tensor reads or challenges
    /// (RFC-0018/0054). Includes fixed-axis nesting, polynomial/coordinate arrays,
    /// shares and read maps, held-read flags, trial coordinate pairs and vector growth.
    /// Borrowed stage/layer-map storage, inline data, payloads and allocator overhead
    /// are additional. This sizes construction; it does not validate a grouping or
    /// authenticate the caller's pairing of inputs. The constructor still validates it.
    /// Term equality must allocate nothing, as for compiler term types; allocations
    /// or callback work introduced by caller-defined equality remain additional.
    ///
    /// # Errors
    /// [`LeafError::Memory`] on checked overflow or an unaddressable array request.
    pub fn construction_memory<T: Copy + Eq>(
        stage: &LeafLayout<T>,
        groups: LeafGroups,
    ) -> Result<GroupLayoutMemory, LeafError> {
        let mut bytes = array::<Vec<Vec<Option<u32>>>>(stage.sites().len() as u64)?;
        bytes = add(bytes, array::<TableId>(stage.instances().len() as u64)?)?;
        let mut shares = 0;
        let mut polynomials = 0;
        let mut flags = 0;
        let mut trial = 0;
        for (site_id, site) in stage.sites().iter().enumerate() {
            shares = add(shares, site.layers.len() as u64)?;
            bytes = add(bytes, array::<Vec<Option<u32>>>(site.layers.len() as u64)?)?;
            for (layer, gadget) in site.layers.iter().enumerate() {
                let lookups = gadget.lookups.len() as u64;
                bytes = add(bytes, array::<Option<u32>>(lookups)?)?;
                bytes = add(bytes, array::<(TableId, Block, bool)>(lookups)?)?;
                let reads = stage.reads(site_id, layer).ok_or(LeafError::Internal {
                    rule: "the layout reads every site's tensors",
                })?;
                let count = reads.len() as u64;
                polynomials = add(polynomials, count)?;
                // Each public map is preallocated from held membership before filling.
                bytes = add(bytes, mul(array::<usize>(count)?, 2)?)?;
                flags = flags.max(array::<(bool, bool)>(count)?);
                for read in reads {
                    let coords = read.coords.len() as u64;
                    let vars = coords.min(u64::from(groups.token_bits));
                    let point = add(array::<u32>(coords)?, pushed::<usize>(vars)?)?;
                    bytes = add(bytes, point)?;
                    // The first matching lookup's pair is retained; a trial pair and
                    // its own growth coexist while checking another matching lookup.
                    trial = trial.max(add(point, old_capacity::<usize>(vars)?)?);
                }
            }
        }
        bytes = add(bytes, array::<MergePolynomial>(polynomials)?)?;
        bytes = add(bytes, array::<Vec<u32>>(polynomials)?)?;
        bytes = add(bytes, array::<Share>(shares)?)?;
        Ok(GroupLayoutMemory {
            output: bytes,
            peak: add(add(bytes, flags)?, trial)?,
        })
    }

    /// Requested peak heap bytes of [`Self::group`], including the returned layout.
    ///
    /// Allocation-free public sizing (RFC-0018/0054). Includes read/coordinate
    /// clones, block/instance capacities, B-tree insertion/iteration and stable-sort
    /// scratch. Scans the cached public table ids against held blocks, without a
    /// planning allocation or any assumption about transcript challenges.
    /// Borrowed grouping metadata, stage, payloads, inline data and allocator
    /// overhead remain additional. This does not price proving the group.
    ///
    /// # Errors
    /// The same invalid-group error as [`Self::group`], or [`LeafError::Memory`]
    /// on checked overflow/unaddressable array requests.
    pub fn group_memory(&self, group: usize) -> Result<GroupLayoutMemory, LeafError> {
        if group >= self.count() {
            return super::internal("a group of the grouping");
        }
        let tokens = group as u64 & ((1u64 << self.groups.token_bits) - 1);
        let subset = (group as u64) >> self.groups.token_bits;
        let mut reads = 0;
        let mut points = 0;
        for share in self.shares.iter().filter(|share| share.subset == subset) {
            let held = if tokens == 0 {
                &share.reads_zero
            } else {
                &share.reads
            };
            reads = add(reads, held.len() as u64)?;
            for &read in held {
                points = add(
                    points,
                    array::<u32>(self.coords[share.first + read].len() as u64)?,
                )?;
            }
        }
        let mut bytes = add(pushed::<GroupRead>(reads)?, points)?;
        let mut tables = 0;
        let mut sort = 0;
        let mut growth = 0;
        for &table in &self.tables {
            let mut blocks = 0;
            for share in self.shares.iter().filter(|share| share.subset == subset) {
                for &(id, _, restricted) in &share.blocks {
                    if id == table && (restricted || tokens == 0) {
                        blocks = add(blocks, 1)?;
                    }
                }
            }
            if blocks > 0 {
                tables = add(tables, 1)?;
                // Buffers move from the index to instances without cloning.
                bytes = add(bytes, pushed::<Block>(blocks)?)?;
                sort = sort.max(array::<Block>(blocks)?);
                growth = growth.max(old_capacity::<Block>(blocks)?);
            }
        }
        bytes = add(bytes, array::<InstanceLayout>(tables)?)?;
        let output = bytes;
        bytes = add(bytes, old_capacity::<GroupRead>(reads)?)?;
        // Block-vector growth ends before instance_layouts starts stable sorting.
        bytes = add(bytes, sort.max(growth))?;
        if tables == 0 {
            return Ok(GroupLayoutMemory {
                output,
                peak: bytes,
            });
        }
        // Rust 1.95 B-tree B=6: 11 keys/values, 12 edges, parent and two
        // u16s. Bound both node alignments and transient insertion splits.
        let align = align_of::<TableId>()
            .max(align_of::<Vec<Block>>())
            .max(align_of::<usize>()) as u64;
        let node = add(
            add(
                size_of::<usize>() as u64 + 4,
                11 * (size_of::<TableId>() + size_of::<Vec<Block>>()) as u64,
            )?,
            add(12 * size_of::<usize>() as u64, 7 * (align - 1))?,
        )?;
        Ok(GroupLayoutMemory {
            output,
            peak: add(bytes, mul(mul(add(tables, 1)?, 2)?, node)?)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observe_growth<T: Clone>(value: T) {
        let mut values = Vec::new();
        let mut largest_old = 0;
        for len in 1..=1025 {
            let before = values.capacity();
            values.push(value.clone());
            if values.capacity() != before {
                largest_old = largest_old.max(before);
            }
            assert_eq!(
                pushed::<T>(len).unwrap(),
                (values.capacity() * size_of::<T>()) as u64
            );
            assert_eq!(
                old_capacity::<T>(len).unwrap(),
                (largest_old * size_of::<T>()) as u64
            );
        }
    }

    #[test]
    fn pinned_vector_growth_matches_actual_standard_library_capacities() {
        observe_growth(0u8);
        observe_growth(0usize);
        observe_growth([0u8; 1025]);
        observe_growth(GroupRead {
            site: 0,
            layer: 0,
            read: 0,
            polynomial: 0,
            coords: Vec::new(),
            group_bits: 0,
        });
    }

    #[test]
    fn array_and_growth_overflow_refuse_without_allocating() {
        assert_eq!(array::<u64>(u64::MAX), Err(LeafError::Memory));
        assert_eq!(pushed::<GroupRead>(u64::MAX), Err(LeafError::Memory));
        assert_eq!(old_capacity::<Share>(u64::MAX), Err(LeafError::Memory));
        assert_eq!(add(u64::MAX, 1), Err(LeafError::Memory));
        assert_eq!(mul(u64::MAX, 2), Err(LeafError::Memory));
    }
}
