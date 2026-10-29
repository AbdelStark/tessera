// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Original-source exact-greedy stage tables, from draft RFC-0053's Boolean formulas.

use super::*;
use tessera_compiler::{GreedyCertificate, GreedyKeyColumn, schedule::Deferred};

#[derive(Clone, Copy)]
pub(super) enum Runtime {
    Tokens,
    Key(GreedyCertificate, GreedyKeyColumn),
    Selected(GreedyCertificate),
    Observe,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn append(
    lowering: &Lowering,
    runtime: &RuntimePlan,
    layout: &LeafLayout<ModelColumn>,
    families: &mut Vec<Family>,
    slots: &mut Vec<Slot>,
    native: &mut Vec<Native>,
    deferred: &mut Vec<Deferred>,
) -> Result<(), StageError> {
    let h = lowering.graph.producer(lowering.one_hot).ok_or_else(bad)?;
    let l = lowering.graph.producer(lowering.logits).ok_or_else(bad)?;
    let shape = &lowering.graph.tensors[lowering.logits.0 as usize].shape;
    let &[n, v] = shape.as_slice() else {
        return Err(bad());
    };
    let nt = vars(&[n]);
    let nv = vars(&[v]);
    let mut add = |kind, node, degree, evaluations, zero, outputs, site| {
        let family = families.len();
        families.push(Family {
            windows: if matches!(kind, Runtime::Selected(_)) {
                runtime.key_columns().to_vec()
            } else {
                Vec::new()
            },
            site,
            nodes: vec![node],
            shape: shape.clone(),
            lambda: 0,
            kind: Kind::Runtime(kind),
        });
        native.push(Native {
            tag: (native.len() + 1) as u32,
            kind: match kind {
                Runtime::Tokens => NativeKind::Tokens,
                Runtime::Key(..) => NativeKind::GreedyKey,
                Runtime::Selected(_) => NativeKind::GreedySelected,
                Runtime::Observe => NativeKind::Observe,
            },
            node,
            vars: nt + nv,
            degree,
            evaluations,
            zero,
            outputs,
        });
        slots.push(Slot {
            family,
            relation: Relation::Runtime,
        });
    };
    add(
        Runtime::Tokens,
        h,
        2,
        2,
        true,
        vec![
            NativeOutput {
                tensor: lowering.one_hot,
                from: 0,
                vars: nv + nt,
            },
            NativeOutput {
                tensor: lowering.tokens,
                from: nv,
                vars: nt,
            },
        ],
        0,
    );
    deferred.push((h, Some(0)));
    let outputs = vec![NativeOutput {
        tensor: lowering.logits,
        from: 0,
        vars: nv + nt,
    }];
    if let Some(c) = runtime.certificate() {
        let site = layout
            .sites()
            .iter()
            .enumerate()
            .find_map(|(s, _)| {
                layout
                    .reads(s, 0)?
                    .iter()
                    .any(|r| r.term.is_greedy_key())
                    .then_some(s)
            })
            .ok_or_else(bad)?;
        for &column in runtime.key_columns() {
            add(
                Runtime::Key(c, column),
                l,
                2,
                nt as usize + 3,
                false,
                outputs.clone(),
                site,
            );
        }
        add(
            Runtime::Selected(c),
            l,
            3,
            nt as usize + 3,
            true,
            outputs,
            site,
        );
    } else {
        add(Runtime::Observe, l, 1, 1, true, outputs, 0);
    }
    deferred.push((l, None));
    Ok(())
}

fn graph_column(
    lowering: &Lowering,
    graph: &impl WitnessSource,
    tensor: TensorId,
) -> Result<SecretBuf<G>, StageError> {
    let shape = &lowering.graph.tensors[tensor.0 as usize].shape;
    let mut out = zeros(1usize.checked_shl(vars(shape)).ok_or(StageError::Memory)?)?;
    pad(graph.values(tensor).ok_or_else(bad)?, shape, &mut out)?;
    Ok(out)
}

pub(super) fn build(
    prover: &mut ModelNativeProver<'_>,
    family: &Family,
    kind: Runtime,
    lowering: &Lowering,
    witness: &ModelWitness,
    graph: &impl WitnessSource,
    leaf: &dyn LeafClaims,
) -> Result<(), StageError> {
    let &[n, v] = family.shape.as_slice() else {
        return Err(bad());
    };
    let (nt, nv) = (vars(&[n]), vars(&[v]));
    let (tp, vp) = (1usize << nt, 1usize << nv);
    let mut raw = Vec::new();
    let mut push = |values| raw.push(prover.table(values));
    if matches!(kind, Runtime::Tokens) {
        push(graph_column(lowering, graph, lowering.one_hot)?);
        let tokens = graph_column(lowering, graph, lowering.tokens)?;
        let mut expanded = zeros(tp * vp)?;
        for (row, &token) in expanded.chunks_exact_mut(vp).zip(tokens.iter()) {
            row.fill(token);
        }
        push(expanded);
    } else {
        push(graph_column(lowering, graph, lowering.logits)?);
        if let Runtime::Key(c, _) | Runtime::Selected(c) = kind {
            let (id, m) = witness.greedy_maximum().ok_or_else(bad)?;
            if id != c.gadget || m.len() != tp {
                return Err(bad());
            }
            let mut expanded = zeros(tp * vp)?;
            for (row, &value) in expanded.chunks_exact_mut(vp).zip(m) {
                row.fill(G::from_i64(value));
            }
            push(expanded);
            let h = graph_column(lowering, graph, lowering.one_hot)?;
            for k in 0..=nt {
                let mut image = zeros(tp * vp)?;
                for (t, row) in image.chunks_exact_mut(vp).enumerate() {
                    let original = if k == nt {
                        0
                    } else {
                        (t & !((1usize << (k + 1)) - 1)) | (1usize << k)
                    };
                    row.copy_from_slice(&h[original * vp..(original + 1) * vp]);
                }
                push(image);
            }
        }
    }
    let (point, fixed) = if let Runtime::Key(_, column) = kind {
        // The greedy key has no token axis (RFC-0054 'No token axis'): no group variables.
        let point = leaf.point(column.variables as usize, &[]).ok_or_else(bad)?;
        let p = point.as_slice();
        let keys = witness
            .values(
                family.site,
                0,
                ModelColumn::greedy_key(column.request_index),
            )
            .ok_or_else(bad)?;
        let mut values = zeros(keys.len())?;
        for (out, &key) in values.iter_mut().zip(keys.iter()) {
            *out = G::from_i64(key);
        }
        (p.to_vec(), vec![evaluate(&values, p)?])
    } else {
        (Vec::new(), Vec::new())
    };
    prover.add(raw, None, Vec::new(), fixed, point, true);
    Ok(())
}

pub(super) fn prepare(
    data: &mut Data,
    family: &Family,
    kind: Runtime,
    context: &NativeContext<'_>,
) -> Result<(), StageError> {
    let &[n, v] = family.shape.as_slice() else {
        return Err(bad());
    };
    let (nt, nv) = (vars(&[n]) as usize, vars(&[v]) as usize);
    let (tp, vp) = (1usize << nt, 1usize << nv);
    let zero = context
        .zero
        .get(context.zero.len().checked_sub(nt + nv).ok_or_else(bad)?..)
        .ok_or_else(bad)?;
    let time = eq_table(&zero[nv..]).map_err(|_| bad())?;
    data.public.clear();
    data.terms.clear();
    data.equality = Default::default();
    data.target = F::ZERO;
    let count = data.raw.len();
    let mut coefficients = vec![
        vec![F::ZERO; tp * vp];
        if matches!(kind, Runtime::Key(..)) {
            3
        } else {
            2
        }
    ];
    if matches!(kind, Runtime::Key(..) | Runtime::Selected(_)) {
        let (vocabulary, outputs) = if matches!(kind, Runtime::Key(..)) {
            (
                eq_table(&data.point[..nv]).map_err(|_| bad())?,
                eq_table(&data.point[nv..]).map_err(|_| bad())?,
            )
        } else {
            (eq_table(&zero[..nv]).map_err(|_| bad())?, Vec::new())
        };
        let mass = vocabulary
            .iter()
            .take(v as usize)
            .copied()
            .fold(F::ZERO, |a, x| a + x);
        for t in 0..tp {
            let window = match kind {
                Runtime::Key(_, column) => Some(column),
                _ => family.windows.iter().copied().find(|column| {
                    let start = (column.slot_offset + column.prompt - 1) as usize;
                    let end = (column.slot_offset + column.positions - 1) as usize;
                    start <= t && t < end
                }),
            };
            let start = window.map_or(0, |column| {
                (column.slot_offset + column.prompt - 1) as usize
            });
            let end = window.map_or(0, |column| {
                (column.slot_offset + column.positions - 1) as usize
            });
            let active = start <= t && t < end;
            let weight = if matches!(kind, Runtime::Key(..)) {
                if active { outputs[t - start] } else { F::ZERO }
            } else {
                time[t]
            };
            let mut suffix = F::ZERO;
            for (j, &vocabulary_weight) in vocabulary.iter().enumerate() {
                let i = t * vp + j;
                if matches!(kind, Runtime::Key(..)) {
                    coefficients[0][i] = if j == 0 { weight * mass } else { F::ZERO };
                    if j < v as usize {
                        coefficients[1][i] = weight * vocabulary_weight;
                        coefficients[2][i] = weight * suffix;
                        suffix += vocabulary_weight;
                    }
                } else {
                    coefficients[0][i] = if j == 0 { weight } else { F::ZERO };
                    coefficients[1][i] = if active && j < v as usize {
                        weight
                    } else {
                        F::ZERO
                    };
                }
            }
        }
        let alpha = coefficients.len();
        for k in 0..=nt {
            coefficients.push(
                (0..tp)
                    .flat_map(|t| {
                        let selected = if k == nt {
                            t == tp - 1
                        } else {
                            t & ((1usize << (k + 1)) - 1) == (1usize << k) - 1
                        };
                        core::iter::repeat_n(if selected { F::ONE } else { F::ZERO }, vp)
                    })
                    .collect(),
            );
        }
        if matches!(kind, Runtime::Key(..)) {
            data.target = *data.fixed.first().ok_or_else(bad)?;
            data.terms.push(Term::new(F::ONE, vec![1, count]));
            data.terms.push(Term::new(-F::ONE, vec![0, count + 1]));
            for k in 0..=nt {
                data.terms.push(Term::new(
                    -F::ONE,
                    vec![2 + k, count + 2, count + alpha + k],
                ));
            }
        } else {
            data.terms.push(Term::new(-F::ONE, vec![1, count]));
            for k in 0..=nt {
                data.terms.push(Term::new(
                    F::ONE,
                    vec![0, 2 + k, count + 1, count + alpha + k],
                ));
            }
        }
    } else if matches!(kind, Runtime::Tokens) {
        for (t, &weight) in time.iter().enumerate() {
            for j in 0..vp {
                coefficients[0][t * vp + j] = weight * fp(j as u64);
                coefficients[1][t * vp + j] = if j == 0 { weight } else { F::ZERO };
            }
        }
        data.terms.push(Term::new(F::ONE, vec![0, count]));
        data.terms.push(Term::new(-F::ONE, vec![1, count + 1]));
    } else {
        coefficients.clear();
        data.terms.push(Term::new(F::ZERO, vec![0]));
    }
    data.public = coefficients;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera_field::QuadraticExtension;
    use tessera_ir::{GadgetId, IntRange, TableId};

    #[test]
    fn batch_key_coefficients_stay_in_own_slot_and_selected_covers_all_windows() {
        let c = GreedyCertificate {
            gadget: GadgetId(1),
            table: TableId(1),
            positions: 12,
            vocab: 2,
            prompt: 1,
            logits: IntRange { lo: -8, hi: 7 },
            range_bits: 4,
            variables: 1,
        };
        let windows = vec![
            GreedyKeyColumn {
                request_index: 0,
                prompt: 1,
                positions: 2,
                slot_offset: 0,
                variables: 1,
            },
            GreedyKeyColumn {
                request_index: 2,
                prompt: 1,
                positions: 3,
                slot_offset: 8,
                variables: 2,
            },
        ];
        let zero = vec![F::from(G::from_u64_reduced(3)); 5];
        for kind in [Runtime::Key(c, windows[1]), Runtime::Selected(c)] {
            let family = Family {
                windows: windows.clone(),
                site: 0,
                nodes: Vec::new(),
                shape: vec![12, 2],
                lambda: 0,
                kind: Kind::Runtime(kind),
            };
            let mut data = Data {
                score: None,
                raw: vec![0; 7],
                mask: None,
                terms: Vec::new(),
                fixed: vec![F::ZERO],
                point: vec![F::from(G::from_u64_reduced(2)); 2],
                full_point: true,
                equality: Default::default(),
                target: F::ZERO,
                public: Vec::new(),
            };
            prepare(
                &mut data,
                &family,
                kind,
                &NativeContext {
                    stage: 1,
                    layer: &[],
                    zero: &zero,
                },
            )
            .unwrap();
            for t in 0..16 {
                for v in 0..2 {
                    let expected = match kind {
                        Runtime::Key(..) => {
                            let output_weight = match t {
                                8 => F::ONE - data.point[1],
                                9 => data.point[1],
                                _ => F::ZERO,
                            };
                            output_weight
                                * if v == 0 {
                                    F::ONE - data.point[0]
                                } else {
                                    data.point[0]
                                }
                        }
                        _ => {
                            let active = matches!(t, 0 | 8 | 9);
                            if active {
                                zero[1..].iter().enumerate().fold(F::ONE, |acc, (bit, &r)| {
                                    acc * if t >> bit & 1 == 1 { r } else { F::ONE - r }
                                })
                            } else {
                                F::ZERO
                            }
                        }
                    };
                    assert_eq!(data.public[1][t * 2 + v], expected, "t={t}, v={v}");
                }
            }
        }
    }

    #[test]
    fn production_coefficients_match_all_literal_native_finals() {
        let vectors: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../spec/vectors/greedy-boundary/greedy-boundary.json"
        ))
        .unwrap();
        let ext = |v: &serde_json::Value| {
            F::new(
                G::from_canonical_u64(v[0].as_str().unwrap().parse().unwrap()).unwrap(),
                G::from_canonical_u64(v[1].as_str().unwrap().parse().unwrap()).unwrap(),
            )
        };
        let point =
            |v: &serde_json::Value| v.as_array().unwrap().iter().map(ext).collect::<Vec<_>>();
        for case in vectors["cases"].as_array().unwrap() {
            let n = case["n"].as_u64().unwrap() as u32;
            let v = case["vocab"].as_u64().unwrap() as u32;
            let prompt = case["prompt"].as_u64().unwrap() as u32;
            let c = GreedyCertificate {
                gadget: GadgetId(1),
                table: TableId(1),
                positions: n,
                vocab: v,
                prompt,
                logits: IntRange { lo: -128, hi: 127 },
                range_bits: 8,
                variables: vars(&[n - prompt, v]),
            };
            let column = GreedyKeyColumn {
                request_index: 0,
                prompt,
                positions: n,
                slot_offset: 0,
                variables: c.variables,
            };
            let p = point(&case["point"]);
            let mut z = vec![F::ZERO; vars(&[v]) as usize];
            z.extend(point(&case["zero"]));
            for (kind, raw, expected) in [
                (Runtime::Tokens, "token_raw", "token_final"),
                (Runtime::Key(c, column), "raw", "key_final"),
                (Runtime::Selected(c), "raw", "selected_final"),
            ] {
                let raw = point(&case[raw]);
                let family = Family {
                    windows: vec![column],
                    site: 0,
                    nodes: Vec::new(),
                    shape: vec![n, v],
                    lambda: 0,
                    kind: Kind::Runtime(kind),
                };
                let mut target = point(&case["target_v"]);
                target.extend(point(&case["target_o"]));
                let mut data = Data {
                    score: None,
                    raw: vec![0; raw.len()],
                    mask: None,
                    terms: Vec::new(),
                    fixed: vec![ext(&case["key_claim"])],
                    point: target,
                    full_point: true,
                    equality: Default::default(),
                    target: F::ZERO,
                    public: Vec::new(),
                };
                prepare(
                    &mut data,
                    &family,
                    kind,
                    &NativeContext {
                        stage: 1,
                        layer: &[],
                        zero: &z,
                    },
                )
                .unwrap();
                let public = data
                    .public
                    .iter()
                    .map(|v| tessera_poly::evaluate(v, &p).unwrap());
                let values = raw.iter().copied().chain(public).collect::<Vec<_>>();
                let actual = data.terms.iter().fold(F::ZERO, |sum, term| {
                    sum + *term.weight() * term.factors().iter().fold(F::ONE, |p, &i| p * values[i])
                });
                assert_eq!(
                    actual,
                    ext(&case[expected]),
                    "n={n} v={v} prompt={prompt} {expected}"
                );
            }
        }
    }
}
