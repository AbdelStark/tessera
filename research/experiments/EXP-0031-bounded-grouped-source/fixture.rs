// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Public synthetic descriptors and deterministic values, never real model inputs.

use tessera_codec::SecretBuf;
use tessera_compiler::schedule::{LayerMap, NodeRef};
use tessera_field::{Field, Goldilocks as G, PrimeField};
use tessera_gadgets::{Affine, Lookup};
use tessera_ir::NodeId;
use tessera_prover::leaf::{LeafLayout, Read, Site, SiteGadget, SiteLookup, SiteTensors};
use tessera_tables::TableId;

pub const SEED: u64 = 0x5eed0031;
pub const LENGTHS: [&[u32]; 6] = [&[1], &[3], &[5], &[2, 1], &[3, 2], &[5, 1, 3]];
pub const LAYERS: [u32; 6] = [2, 3, 5, 2, 3, 5];
pub const GROUPS: [(u8, u8); 4] = [(0, 1), (1, 0), (1, 1), (2, 1)];

pub fn bits(n: u32) -> u32 {
    n.next_power_of_two().trailing_zeros()
}

pub fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn affine(constant: u64, terms: &[(u64, u8)]) -> Affine<G, u8> {
    Affine {
        constant: G::from_u64_reduced(constant),
        terms: terms
            .iter()
            .map(|&(c, t)| (G::from_u64_reduced(c), t))
            .collect(),
    }
}

fn lookup(
    domain: Vec<u32>,
    token: Option<usize>,
    x: u8,
    class: Option<(u8, Vec<usize>)>,
    valued: bool,
) -> SiteLookup<u8> {
    let mut terms = vec![(1, x)];
    if let Some((c, _)) = &class {
        terms.push((4, *c));
    }
    let axes = class.into_iter().collect();
    SiteLookup {
        lookup: Lookup {
            table: TableId(if valued { 2 } else { 1 }),
            key: affine(0, &terms),
            value: valued.then(|| {
                affine(
                    3,
                    &terms.iter().map(|&(c, t)| (2 * c, t)).collect::<Vec<_>>(),
                )
            }),
        },
        domain,
        axes,
        token,
    }
}

pub struct Fixture {
    pub id: usize,
    pub lengths: Vec<u32>,
    pub slot: u32,
    pub tokens: u32,
    pub layers: LayerMap,
    pub stage: LeafLayout<u8>,
}

impl Fixture {
    pub fn new(id: usize) -> Self {
        let lengths = LENGTHS[id].to_vec();
        let slot = lengths
            .iter()
            .copied()
            .max()
            .unwrap_or(1)
            .next_power_of_two();
        // RFC-0071: one request's unpadded axis is n; several slots use R*S.
        let tokens = if lengths.len() == 1 {
            lengths[0]
        } else {
            lengths.len() as u32 * slot
        };
        let mut layers = LayerMap::none();
        // Hand-built synthetic map; private graph-kind fields are unused by leaf layouts.
        layers.count = LAYERS[id];
        layers.vars = bits(layers.count);
        layers.first = 1;
        layers.width = 1;
        let gadget = SiteGadget {
            lookups: vec![
                lookup(vec![tokens, 3], Some(0), 0, Some((1, vec![1])), false),
                lookup(vec![2, tokens, 3], Some(1), 3, Some((4, vec![0, 2])), true),
                lookup(vec![3], None, 2, None, false),
            ],
        };
        let sites = vec![
            Site {
                node: NodeRef::Node(NodeId(0)),
                layers: vec![SiteGadget {
                    lookups: vec![lookup(vec![5], None, 5, None, false)],
                }],
            },
            Site {
                node: NodeRef::Position(0),
                layers: vec![gadget; layers.count as usize],
            },
            Site {
                node: NodeRef::Node(NodeId(layers.count + 1)),
                layers: vec![SiteGadget {
                    lookups: vec![lookup(vec![tokens, 2], Some(0), 6, None, true)],
                }],
            },
        ];
        let stage = LeafLayout::new(&layers, sites).expect("registered synthetic layout");
        Self {
            id,
            lengths,
            slot,
            tokens,
            layers,
            stage,
        }
    }

    pub fn real_token(&self, token: u64) -> bool {
        let request = token / u64::from(self.slot);
        let position = token % u64::from(self.slot);
        self.lengths
            .get(request as usize)
            .is_some_and(|&n| position < u64::from(n))
    }

    pub fn read(&self, site: usize, layer: usize, term: u8) -> &Read<u8> {
        self.stage
            .reads(site, layer)
            .and_then(|reads| reads.iter().find(|r| r.term == term))
            .expect("registered source identity")
    }

    pub fn datum(&self, site: usize, layer: usize, term: u8, index: u64) -> i64 {
        let read = self.read(site, layer, term);
        let lookup = self.stage.sites()[site].layers[layer]
            .lookups
            .iter()
            .find(|l| {
                l.lookup
                    .key
                    .terms
                    .iter()
                    .chain(l.lookup.value.iter().flat_map(|v| &v.terms))
                    .any(|&(_, t)| t == term)
            })
            .expect("source has lookup");
        let mut remaining = index;
        for &axis in read.axes.iter().rev() {
            let width = bits(lookup.domain[axis]);
            let at = remaining & ((1 << width) - 1);
            remaining >>= width;
            if at >= u64::from(lookup.domain[axis]) {
                return 0;
            }
            if lookup.token == Some(axis) && !self.real_token(at) {
                return 0;
            }
        }
        assert_eq!(remaining, 0, "source index within declared tensor");
        (splitmix(
            SEED ^ ((self.id as u64) << 48)
                ^ ((site as u64) << 40)
                ^ ((layer as u64) << 32)
                ^ (u64::from(term) << 24)
                ^ index,
        ) & 3) as i64
    }

    pub fn materialize(&self) -> Resident {
        self.stage
            .sites()
            .iter()
            .enumerate()
            .map(|(s, site)| {
                (0..site.layers.len())
                    .map(|l| {
                        self.stage
                            .reads(s, l)
                            .expect("reads")
                            .iter()
                            .map(|read| {
                                let n = 1usize << read.coords.len();
                                let mut values =
                                    SecretBuf::try_with_capacity(n).expect("bounded fixture");
                                for i in 0..n {
                                    values.push(self.datum(s, l, read.term, i as u64));
                                }
                                (read.term, values)
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}

pub type Resident = Vec<Vec<Vec<(u8, SecretBuf<i64>)>>>;

pub fn views(resident: &Resident) -> Vec<Vec<SiteTensors<'_, u8>>> {
    resident
        .iter()
        .map(|layers| {
            layers
                .iter()
                .map(|terms| SiteTensors::new(terms.iter().map(|(t, v)| (*t, &v[..])).collect()))
                .collect()
        })
        .collect()
}

pub struct ForestGeometry {
    pub height: u32,
    pub queries: u32,
    pub layers: u32,
    pub vars: u32,
}

impl ForestGeometry {
    pub fn new(f: &Fixture, height: u32) -> Self {
        let queries = height;
        let layers = queries + bits(f.tokens);
        Self {
            height,
            queries,
            layers,
            vars: layers + f.layers.vars + 1,
        }
    }

    pub fn datum(&self, fixture: &Fixture, index: u64) -> G {
        let token = (index >> self.queries) & ((1 << (self.layers - self.queries)) - 1);
        let layer = (index >> self.layers) & ((1 << fixture.layers.vars) - 1);
        if !fixture.real_token(token) || layer >= u64::from(fixture.layers.count) {
            G::ONE
        } else {
            G::from_u64_reduced(1 + splitmix(SEED ^ index ^ ((fixture.id as u64) << 48)) % 3)
        }
    }

    pub fn materialize(&self, fixture: &Fixture) -> SecretBuf<G> {
        let mut out = SecretBuf::try_with_capacity(1 << self.vars).expect("bounded forest");
        for x in 0..1 << self.vars {
            out.push(self.datum(fixture, x));
        }
        out
    }
}
