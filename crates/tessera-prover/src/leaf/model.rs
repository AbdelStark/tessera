// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Current-model lookup sites from the scalar gadgets' canonical constraints.
use super::layout::{token, token_rules};
use super::{LeafError, LeafLayout, Site, SiteGadget, SiteLookup, gadget_nodes, internal};
use tessera_compiler::schedule::{LayerMap, NodeRef};
use tessera_compiler::{
    AuxColumn, CompareColumn as Array, Gadget, Lowering, RuntimePlan, Selection,
};
use tessera_field::{Field, Goldilocks, PrimeField};
use tessera_gadgets::{
    Affine, Bracket, Committed, CompareColumn, Lookup, Requant, RsqrtColumn, RsqrtRow, RsqrtTerm,
    Side, Slack, SoftmaxColumn, SoftmaxRow, SoftmaxTerm, SoftmaxValue, Term,
};
use tessera_ir::{ClaimList, Node, NodeId, Op};
use tessera_tables::TableKind;

/// A local source of a model gadget's lookup forms. Virtual columns are reduction debts,
/// not new advice positions (RFC-0009/0045/0050).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModelColumn {
    /// The node's original graph input at this index.
    Input(u8),
    /// The node's graph output at this index.
    Output(u8),
    /// A real RFC-0050 auxiliary at this node.
    Auxiliary(AuxColumn),
    /// Root's virtual `u = psi * (SumSquares + epsilon)`.
    RootInput,
    /// Softmax's virtual masked lookup key `kappa`.
    ExpKey,
    /// Softmax's virtual bracket slack, A then B.
    Slack(Slack),
    /// Exact-greedy comparison key, reduced to original model/certificate sources.
    GreedyKey,
}

fn own<T>() -> Result<T, LeafError> {
    internal("model leaf sources and requests follow the canonical lowering and selection")
}
fn map_form<T: Copy>(
    form: Affine<Goldilocks, T>,
    mut column: impl FnMut(T) -> Result<ModelColumn, LeafError>,
) -> Result<Affine<Goldilocks, ModelColumn>, LeafError> {
    Ok(Affine {
        constant: form.constant,
        terms: form
            .terms
            .into_iter()
            .map(|(a, t)| Ok((a, column(t)?)))
            .collect::<Result<_, LeafError>>()?,
    })
}
fn requant(term: Term) -> Result<ModelColumn, LeafError> {
    Ok(match term {
        Term::Input => ModelColumn::Input(0),
        Term::Class => ModelColumn::Input(1),
        Term::Committed(Committed::Y) => ModelColumn::Output(0),
        Term::Committed(c) => ModelColumn::Auxiliary(match c {
            Committed::ZLo => AuxColumn::Low,
            Committed::H0 => AuxColumn::Middle,
            Committed::G => AuxColumn::Saturation,
            _ => return own(),
        }),
    })
}
fn root(term: RsqrtTerm) -> Result<ModelColumn, LeafError> {
    Ok(match term {
        RsqrtTerm::Input => ModelColumn::RootInput,
        RsqrtTerm::Column(RsqrtColumn::Root) => ModelColumn::Output(0),
        RsqrtTerm::Column(RsqrtColumn::InputLimb(j)) => {
            ModelColumn::Auxiliary(AuxColumn::InputLimb(u32::from(j)))
        }
        RsqrtTerm::Column(RsqrtColumn::RootLimb(j)) => {
            ModelColumn::Auxiliary(AuxColumn::RootByte(u32::from(j)))
        }
        RsqrtTerm::Column(RsqrtColumn::Compare(bracket, c)) => {
            let (column, limb) = match c {
                CompareColumn::Digit(Side::Left, j) => (Array::LeftByte, j),
                CompareColumn::Carry(Side::Left, j) => (Array::LeftCarry, j),
                CompareColumn::Digit(Side::Right, j) => (Array::RightByte, j),
                CompareColumn::Carry(Side::Right, j) => (Array::RightCarry, j),
                CompareColumn::Difference(j) => (Array::Difference, j),
                CompareColumn::Borrow(j) => (Array::Borrow, j),
                _ => return own(),
            };
            ModelColumn::Auxiliary(AuxColumn::Comparison {
                upper: bracket == Bracket::Upper,
                column,
                limb: u32::try_from(limb).map_err(|_| LeafError::Internal {
                    rule: "comparison limb index fits u32",
                })?,
            })
        }
        _ => return own(),
    })
}
fn softmax(term: SoftmaxTerm) -> Result<ModelColumn, LeafError> {
    Ok(match term {
        SoftmaxTerm::Virtual(SoftmaxValue::Key) => ModelColumn::ExpKey,
        SoftmaxTerm::Virtual(SoftmaxValue::Slack(slack)) => ModelColumn::Slack(slack),
        SoftmaxTerm::Column(c) => match c {
            SoftmaxColumn::Exponentials => ModelColumn::Output(0),
            SoftmaxColumn::Reciprocal => ModelColumn::Output(1),
            SoftmaxColumn::Maximum => ModelColumn::Auxiliary(AuxColumn::Maximum),
            SoftmaxColumn::Limb(slack, limb) => ModelColumn::Auxiliary(AuxColumn::Slack {
                second: slack == Slack::B,
                limb: u32::from(limb),
            }),
        },
        _ => return own(),
    })
}
fn shape(lowering: &Lowering, tensor: tessera_ir::TensorId) -> Result<&[u32], LeafError> {
    lowering
        .graph
        .tensors
        .get(tensor.0 as usize)
        .map(|t| t.shape.as_slice())
        .ok_or(LeafError::Internal {
            rule: "a model leaf tensor has a public shape",
        })
}
fn map_lookup<T: Copy>(
    lookup: Lookup<Goldilocks, T>,
    domain: &[u32],
    token: Option<usize>,
    column: fn(T) -> Result<ModelColumn, LeafError>,
) -> Result<SiteLookup<ModelColumn>, LeafError> {
    Ok(SiteLookup {
        lookup: Lookup {
            table: lookup.table,
            key: map_form(lookup.key, column)?,
            value: lookup.value.map(|v| map_form(v, column)).transpose()?,
        },
        domain: domain.to_vec(),
        axes: Vec::new(),
        token,
    })
}
/// The site's gadget at `node`, each lookup's token axis its domain tensor's among `tokens`.
fn layer(
    lowering: &Lowering,
    selection: &Selection,
    claims: &ClaimList,
    tokens: &[Option<u8>],
    node: &Node,
) -> Result<SiteGadget<ModelColumn>, LeafError> {
    let (_, ref gadget) = selection.of(node.id).ok_or(LeafError::Internal {
        rule: "every model lookup node has a selected gadget",
    })?;
    let lookups = match node.op {
        Op::Requant { .. } => {
            let shape = shape(lowering, node.inputs[0])?;
            let axis = token(tokens, node.inputs[0]);
            SiteGadget::requant(&Requant::new(gadget, &selection.tables)?, shape, axis)
                .lookups
                .into_iter()
                .map(|l| {
                    let mut out = map_lookup(l.lookup, &l.domain, l.token, requant)?;
                    out.axes = l
                        .axes
                        .into_iter()
                        .map(|(t, a)| Ok((requant(t)?, a)))
                        .collect::<Result<_, LeafError>>()?;
                    Ok(out)
                })
                .collect::<Result<Vec<_>, LeafError>>()?
        }
        Op::TableMap { table } => {
            if *gadget != (Gadget::Lookup { table }) {
                return own();
            }
            let spec =
                selection
                    .tables
                    .iter()
                    .find(|t| t.id == table)
                    .ok_or(LeafError::Internal {
                        rule: "the model leaf table exists",
                    })?;
            let TableKind::Silu {
                b, ref r_by_class, ..
            } = spec.kind
            else {
                return own();
            };
            let domain = shape(lowering, node.inputs[0])?;
            let mut terms = vec![(Goldilocks::ONE, ModelColumn::Input(0))];
            let mut axes = Vec::new();
            if r_by_class.len() > 1 {
                let class = node.inputs.get(1).and_then(|&t| shape(lowering, t).ok());
                if class != domain.last().map(core::slice::from_ref) {
                    return own();
                }
                terms.push((
                    Goldilocks::from_u64_reduced(1u64.checked_shl(u32::from(b)).ok_or(
                        LeafError::Internal {
                            rule: "SiLU's packed class coefficient fits u64",
                        },
                    )?),
                    ModelColumn::Input(1),
                ));
                axes.push((ModelColumn::Input(1), vec![domain.len().saturating_sub(1)]));
            }
            vec![SiteLookup {
                domain: domain.to_vec(),
                axes,
                token: token(tokens, node.inputs[0]),
                lookup: Lookup {
                    table,
                    key: Affine {
                        constant: Goldilocks::from_u64_reduced(
                            1u64.checked_shl(u32::from(b).checked_sub(1).ok_or(
                                LeafError::Internal {
                                    rule: "SiLU has a positive key width",
                                },
                            )?)
                            .ok_or(LeafError::Internal {
                                rule: "SiLU's key offset fits u64",
                            })?,
                        ),
                        terms,
                    },
                    value: Some(Affine {
                        constant: Goldilocks::ZERO,
                        terms: vec![(Goldilocks::ONE, ModelColumn::Output(0))],
                    }),
                },
            }]
        }
        Op::RsqrtRow {
            dim,
            kappa,
            eps_int,
        } => {
            if *gadget != Gadget::RsqrtRow {
                return own();
            }
            let gadget = RsqrtRow::new(dim, kappa, eps_int, &selection.tables)?;
            gadget
                .constraints::<Goldilocks>()?
                .lookups
                .into_iter()
                .map(|l| {
                    let rows = node.outputs[0];
                    map_lookup(l, shape(lowering, rows)?, token(tokens, rows), root)
                })
                .collect::<Result<_, LeafError>>()?
        }
        Op::SoftmaxRow { exp_table, .. } => {
            let Gadget::SoftmaxRow { exp, .. } = *gadget else {
                return own();
            };
            if exp != exp_table {
                return own();
            }
            let gadget = SoftmaxRow::new(
                exp,
                claims.gadget_policy.limb_bits,
                claims.arch.max_context as usize,
                &selection.tables,
            )?;
            gadget
                .constraints::<Goldilocks>()
                .lookups
                .into_iter()
                .map(|l| {
                    // A score's lookups run over the scores, a row's over the reciprocals.
                    let tensor = match l.per_position {
                        true => node.inputs[0],
                        false => node.outputs[1],
                    };
                    map_lookup(
                        Lookup {
                            table: l.table,
                            key: l.key,
                            value: l.value,
                        },
                        shape(lowering, tensor)?,
                        token(tokens, tensor),
                        softmax,
                    )
                })
                .collect::<Result<_, LeafError>>()?
        }
        _ => return own(),
    };
    if !lookups.iter().map(|l| l.lookup.table).eq(selection
        .reads(node.id)
        .ok_or(LeafError::Internal {
            rule: "the model selection has ordered lookup requests",
        })?
        .iter()
        .copied())
    {
        return own();
    }
    Ok(SiteGadget { lookups })
}

impl LeafLayout<ModelColumn> {
    /// Canonical current-model lookup sites in RFC-0045 order, using only public metadata.
    /// The inputs must describe the same validated lowering, selection and claim list.
    /// OneHot contributes no lookup. All root/Softmax virtuals retain native reduction debts.
    ///
    /// Each lookup's token axis is RFC-0054's, which only a leaf stage in token groups reads. A
    /// graph whose token axes RFC-0054 does not derive, such as one holding RFC-0066's
    /// operations, gets a layout whose lookups have none, which runs ungrouped
    /// ([`LeafLayout::ungroupable`], RFC-0072 'Non-Goals' 'Token groups').
    ///
    /// # Errors
    /// `Internal` for inconsistent own metadata, `Gadget` for invalid scalar constraints,
    /// and the shape/source errors of `LeafLayout::new`.
    pub fn model(
        lowering: &Lowering,
        layers: &LayerMap,
        selection: &Selection,
        claims: &ClaimList,
    ) -> Result<Self, LeafError> {
        if claims.tables != selection.tables {
            return own();
        }
        Self::derive_model(lowering, layers, selection, claims, None)
    }

    /// Complete model sites followed by the request's exact-greedy key (draft RFC-0053).
    ///
    /// # Errors
    /// The same public geometry and scalar constraint errors as [`Self::model`].
    pub fn runtime(
        lowering: &Lowering,
        layers: &LayerMap,
        runtime: &RuntimePlan,
        claims: &ClaimList,
    ) -> Result<Self, LeafError> {
        if !runtime.selection().tables.starts_with(&claims.tables) {
            return own();
        }
        Self::derive_model(
            lowering,
            layers,
            runtime.selection(),
            claims,
            runtime.certificate(),
        )
    }

    fn derive_model(
        lowering: &Lowering,
        layers: &LayerMap,
        selection: &Selection,
        claims: &ClaimList,
        greedy: Option<tessera_compiler::GreedyCertificate>,
    ) -> Result<Self, LeafError> {
        // Token axes only a grouped leaf stage reads (RFC-0072 'Non-Goals' 'Token groups').
        let (tokens, ungroupable) = match token_rules(lowering) {
            Ok(tokens) => (tokens, None),
            Err(rule) => (Vec::new(), Some(rule)),
        };
        let mut sites = Vec::new();
        for site in gadget_nodes(lowering, layers) {
            let nodes: Vec<NodeId> = match site {
                NodeRef::Node(id) => vec![id],
                NodeRef::Position(p) => (0..layers.count).map(|l| layers.node(p, l)).collect(),
            };
            if matches!(
                lowering.graph.nodes[nodes[0].0 as usize].op,
                Op::OneHot { .. }
            ) {
                continue;
            }
            sites.push(Site {
                node: site,
                layers: nodes
                    .into_iter()
                    .map(|id| {
                        layer(
                            lowering,
                            selection,
                            claims,
                            &tokens,
                            &lowering.graph.nodes[id.0 as usize],
                        )
                    })
                    .collect::<Result<_, LeafError>>()?,
            });
        }
        if let Some(c) = greedy {
            let node = lowering
                .graph
                .nodes
                .iter()
                .find(|node| node.outputs.contains(&lowering.logits))
                .ok_or(LeafError::Internal {
                    rule: "the runtime key belongs to the actual LM-head node",
                })?
                .id;
            sites.push(Site {
                node: NodeRef::Node(node),
                layers: vec![SiteGadget {
                    lookups: vec![SiteLookup {
                        lookup: Lookup {
                            table: c.table,
                            key: Affine {
                                constant: Goldilocks::ZERO,
                                terms: vec![(Goldilocks::ONE, ModelColumn::GreedyKey)],
                            },
                            value: None,
                        },
                        domain: vec![c.positions - c.prompt, c.vocab],
                        axes: Vec::new(),
                        // The generated positions, not the batch's token index: no token
                        // indexes the key, so its block stays in token subset 0 (RFC-0054
                        // 'What a group holds').
                        token: None,
                    }],
                }],
            });
        }
        match ungroupable {
            None => Self::new(layers, sites),
            Some(rule) => Self::ungrouped(layers, sites, rule),
        }
    }
}
