// SPDX-License-Identifier: Apache-2.0 OR MIT

//! CPU mixed-boundary sumchecks with individually bound original endpoints.
//! These synthetic stage components are not complete model proofs or Live sessions.

use serde_json::Value;
use std::num::NonZeroUsize;
use tessera_codec::{Ascii, Secret, encode};
use tessera_field::{Field, Goldilocks as G, GoldilocksExt2 as F, PrimeField, QuadraticExtension};
use tessera_hash::{Mode, ParamSetId, SessionContext, Transcript};
use tessera_pcs as pcs;
use tessera_sumcheck::{Instance, MAX_DEGREE, Prefix, Table, Term, prove_batch};
use tv_core::claims::{ClaimList, QuantPoint};
use tv_core::gadgets::{
    Plan,
    greedy::{Kind, Source},
};
use tv_core::{
    lowering, opening, sumcheck as tv,
    validation::{self, GadgetId},
};
use tv_field::{Fp, Fp2};
use tv_hash::transcript as vt;

fn json(path: &str) -> Value {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/vectors");
    serde_json::from_slice(&std::fs::read(root.join(path)).unwrap()).unwrap()
}
fn number(v: &Value) -> u64 {
    v.as_u64()
        .unwrap_or_else(|| v.as_str().unwrap().parse().unwrap())
}
fn extension(v: &Value) -> F {
    F::new(
        G::from_canonical_u64(number(&v[0])).unwrap(),
        G::from_canonical_u64(number(&v[1])).unwrap(),
    )
}
fn fields(v: &Value) -> Vec<F> {
    v.as_array().unwrap().iter().map(extension).collect()
}
fn bases(v: &Value) -> Vec<G> {
    fields(v)
        .into_iter()
        .map(|f| {
            let [a, b] = f.coeffs();
            assert_eq!(b, G::ZERO);
            a
        })
        .collect()
}
fn independent(v: F) -> Fp2 {
    let [a, b] = v.coeffs();
    Fp2::new(
        Fp::from_canonical(a.to_canonical_u64()).unwrap(),
        Fp::from_canonical(b.to_canonical_u64()).unwrap(),
    )
}
fn producer(v: Fp2) -> F {
    F::new(
        G::from_u64_reduced(v.c0.to_canonical()),
        G::from_u64_reduced(v.c1.to_canonical()),
    )
}
fn eq(p: &[F], i: usize) -> F {
    p.iter().enumerate().fold(F::ONE, |a, (j, &r)| {
        a * if i >> j & 1 == 0 { F::ONE - r } else { r }
    })
}
fn variables(n: usize) -> usize {
    usize::BITS.saturating_sub(n.saturating_sub(1).leading_zeros()) as usize
}

struct Cpu {
    secret: Vec<Secret<Vec<G>>>,
    public: Vec<Vec<F>>,
    terms: Vec<Term<F>>,
}
impl Cpu {
    fn instance(&self) -> Instance<'_, F> {
        Instance::sum_of_products(
            self.secret
                .iter()
                .map(Table::secret_base)
                .chain(self.public.iter().map(|v| Table::ext(v).public()))
                .collect(),
            self.terms.clone(),
        )
    }
}
struct Data {
    l: Vec<G>,
    h: Vec<G>,
    m: Vec<G>,
    tokens: Vec<G>,
    n: usize,
    v: usize,
    prompt: usize,
}
impl Data {
    fn read(c: &Value) -> Self {
        Self {
            l: bases(&c["logits"]),
            h: bases(&c["onehot"]),
            m: bases(&c["maximum"]),
            tokens: bases(&c["tokens"]),
            n: number(&c["n"]) as usize,
            v: number(&c["vocab"]) as usize,
            prompt: number(&c["prompt"]) as usize,
        }
    }
    fn build(&self, kind: Kind, pv: &[F], po: &[F], z: &[F]) -> Cpu {
        let nv = variables(self.v);
        let nt = variables(self.n);
        let vp = 1 << nv;
        let tp = 1 << nt;
        let mut source = if kind == Kind::Tokens {
            vec![
                self.h.clone(),
                (0..tp).flat_map(|t| vec![self.tokens[t]; vp]).collect(),
            ]
        } else {
            vec![
                self.l.clone(),
                (0..tp).flat_map(|t| vec![self.m[t]; vp]).collect(),
            ]
        };
        if kind != Kind::Tokens {
            for k in 0..=nt {
                let partial = (0..tp)
                    .flat_map(|t| {
                        let original = if k == nt {
                            0
                        } else {
                            (t & !((1 << (k + 1)) - 1)) | (1 << k)
                        };
                        self.h[original * vp..(original + 1) * vp].iter().copied()
                    })
                    .collect();
                source.push(partial);
            }
        }
        let count = source.len();
        let mut public = Vec::new();
        let mut terms = Vec::new();
        if kind == Kind::Tokens {
            public.push(
                (0..tp)
                    .flat_map(|t| {
                        (0..vp).map(move |v| {
                            eq(z, t) * F::from(G::from_canonical_u64(v as u64).unwrap())
                        })
                    })
                    .collect(),
            );
            public.push(
                (0..tp)
                    .flat_map(|t| (0..vp).map(move |v| if v == 0 { eq(z, t) } else { F::ZERO }))
                    .collect(),
            );
            terms.push(Term::new(F::ONE, vec![0, count]));
            terms.push(Term::new(-F::ONE, vec![1, count + 1]));
        } else {
            let mut cm = Vec::new();
            let mut cl = Vec::new();
            let mut ch = Vec::new();
            let mv = (0..self.v).fold(F::ZERO, |a, v| a + eq(pv, v));
            for t in 0..tp {
                let active = self.prompt - 1 <= t && t < self.n - 1;
                for v in 0..vp {
                    if kind == Kind::Key {
                        let w = if active {
                            eq(po, t - (self.prompt - 1))
                        } else {
                            F::ZERO
                        };
                        cm.push(if v == 0 { w * mv } else { F::ZERO });
                        cl.push(if v < self.v { w * eq(pv, v) } else { F::ZERO });
                        ch.push(if v < self.v {
                            w * (0..v).fold(F::ZERO, |a, i| a + eq(pv, i))
                        } else {
                            F::ZERO
                        });
                    } else {
                        cm.push(if v == 0 { eq(z, t) } else { F::ZERO });
                        cl.push(if active && v < self.v {
                            eq(z, t)
                        } else {
                            F::ZERO
                        });
                    }
                }
            }
            public.push(cm);
            public.push(cl);
            if kind == Kind::Key {
                public.push(ch);
            }
            let alpha_start = public.len();
            for k in 0..=nt {
                public.push(
                    (0..tp)
                        .flat_map(|t| {
                            let selected = if k == nt {
                                t == tp - 1
                            } else {
                                t & ((1 << (k + 1)) - 1) == (1 << k) - 1
                            };
                            vec![if selected { F::ONE } else { F::ZERO }; vp]
                        })
                        .collect(),
                );
            }
            if kind == Kind::Key {
                terms.push(Term::new(F::ONE, vec![1, count]));
                terms.push(Term::new(-F::ONE, vec![0, count + 1]));
                for k in 0..=nt {
                    terms.push(Term::new(
                        -F::ONE,
                        vec![2 + k, count + 2, count + alpha_start + k],
                    ));
                }
            } else {
                terms.push(Term::new(-F::ONE, vec![1, count]));
                for k in 0..=nt {
                    terms.push(Term::new(
                        F::ONE,
                        vec![0, 2 + k, count + 1, count + alpha_start + k],
                    ));
                }
            }
        }
        Cpu {
            secret: source.into_iter().map(Secret::new).collect(),
            public,
            terms,
        }
    }
    fn source(&self, source: Source, point: &[Fp2]) -> Fp2 {
        let values = match source {
            Source::Logits(_) => &self.l,
            Source::OneHot(_) => &self.h,
            Source::Maximum(_) => &self.m,
            Source::Tokens(_) => &self.tokens,
        };
        assert_eq!(values.len(), 1 << point.len());
        values.iter().enumerate().fold(Fp2::ZERO, |a, (i, v)| {
            a + Fp2::from(Fp::from_canonical(v.to_canonical_u64()).unwrap())
                * point.iter().enumerate().fold(Fp2::ONE, |w, (j, &r)| {
                    w * if i >> j & 1 == 0 { Fp2::ONE - r } else { r }
                })
        })
    }
}

fn context(n: usize, v: usize) -> (ClaimList, lowering::Graph) {
    let cases = json("claims/claim-lists.json");
    let c = cases["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["model"] == "tiny-2-servable.d")
        .unwrap();
    let text = c["hex"].as_str().unwrap();
    let bytes = (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect::<Vec<_>>();
    let mut c = ClaimList::open(&bytes, &tv_codec::DecodeLimits::MESSAGE).unwrap();
    c.arch.d_model = 2;
    c.arch.n_heads = 1;
    c.arch.n_kv_heads = 1;
    c.arch.d_head = 2;
    c.arch.d_ff = 6;
    c.arch.vocab = v as u32;
    c.arch.max_context = n.max(8) as u32;
    c.quant.residual_bits = 8;
    c.quant.rsqrt_kappa = 0;
    c.quant.exp_sigma = 3;
    c.quant.residual_exponent = c
        .quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    for (_, s) in &mut c.quant.shift_class_sets {
        *s = vec![12];
    }
    for (_, s) in &mut c.quant.score_scales {
        *s = tv_codec::Rational::new(1, 1).unwrap();
    }
    let public = validation::check_public(&c).unwrap();
    let g = lowering::lower(&public, n as u32).unwrap();
    (c, g)
}
fn sessions(live: bool, id: u8, inputs: &[F], root: &[u8; 32]) -> (Transcript, vt::Transcript) {
    let sid = [id; 16];
    let mut p = Transcript::new(&SessionContext {
        protocol: Ascii::new("tessera-protocol/1").unwrap(),
        mode: if live {
            Mode::Live
        } else {
            Mode::NonInteractive
        },
        params: ParamSetId(Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x53; 32],
        session_id: sid,
    });
    let mut v = vt::Transcript::new(&vt::SessionContext {
        protocol: vt::Ascii::new("tessera-protocol/1").unwrap(),
        mode: if live {
            vt::Mode::Live
        } else {
            vt::Mode::NonInteractive
        },
        params: vt::ParamSetId(vt::Ascii::new("test-small/1").unwrap()),
        statement_digest: [0x53; 32],
        session_id: sid,
    });
    // Fixture labels only: this is not the protocol/1 PCS/session driver.
    p.absorb("fixture/onehot-root", root);
    v.absorb(
        vt::Ascii::new("fixture/onehot-root").unwrap(),
        &encode(root),
    );
    p.absorb("stage/0/split", &inputs.to_vec());
    v.absorb(
        vt::Ascii::new("stage/0/split").unwrap(),
        &encode(&inputs.to_vec()),
    );
    (p, v)
}

// Exercise original-source authentication with real CPU encoding/hash/Merkle openings.
// This primitive fixture has replayable masks and makes no zero-knowledge claim.
fn open_onehot(
    commitment: &pcs::ReferenceCommitment,
    claims: &[tv_core::gadgets::greedy::Claim],
    transcript: &mut vt::Transcript,
) -> Result<(), opening::OpeningError> {
    let nv = commitment.params().log2_k as usize;
    let claims = claims
        .iter()
        .filter(|c| matches!(c.source, Source::OneHot(_)))
        .collect::<Vec<_>>();
    assert!(!claims.is_empty());
    let gamma = transcript
        .challenge_fp2(vt::Ascii::new("fixture/pcs/gamma").unwrap())
        .unwrap();
    let mut power = Fp2::ONE;
    let group = opening::Group {
        column_point: claims[0].point[..nv].to_vec(),
        claims: claims
            .iter()
            .map(|c| {
                assert_eq!(&c.point[..nv], &claims[0].point[..nv]);
                let claim = opening::Claim {
                    row_point: c.point[nv..].to_vec(),
                    gamma: power,
                    value: c.value,
                };
                power *= gamma;
                claim
            })
            .collect(),
    };
    let xi = (0..commitment.layout().row_vars())
        .map(|_| {
            transcript
                .challenge_fp2(vt::Ascii::new("fixture/pcs/xi").unwrap())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let combined = commitment
        .open(
            &[pcs::ClaimGroup {
                column_point: group.column_point.iter().copied().map(producer).collect(),
                claims: group
                    .claims
                    .iter()
                    .map(|c| pcs::RowClaim {
                        row_point: c.row_point.iter().copied().map(producer).collect(),
                        gamma: producer(c.gamma),
                    })
                    .collect(),
            }],
            &xi.iter().copied().map(producer).collect::<Vec<_>>(),
        )
        .unwrap();
    let combined = combined
        .iter()
        .map(|c| opening::CombinedRow {
            u: c.u.iter().copied().map(independent).collect(),
            mask: c.mask.iter().copied().map(independent).collect(),
        })
        .collect::<Vec<_>>();
    let params = commitment.params();
    let groups = [group];
    let evaluated = opening::check_evaluations(
        opening::Opening {
            params: opening::Params {
                log2_k: params.log2_k,
                log2_inv_rate: params.log2_inv_rate,
                mask_coeffs: params.mask_coeffs as u32,
                row_block: params.row_block.get() as u32,
                commitment_id: params.commitment_id,
            },
            layout: opening::Layout {
                row_vars: commitment.layout().row_vars(),
                segments: commitment.layout().segments(),
            },
            root: commitment.root(),
            groups: &groups,
            xi_point: &xi,
        },
        &combined,
    )?;
    // Query every column so this bounded regression has no sampling miss.
    let queries = (0..commitment.codeword_len() as u64).collect::<Vec<_>>();
    let answer = commitment.answer(&queries).unwrap();
    let columns = answer
        .columns
        .iter()
        .map(|c| {
            c.iter()
                .map(|v| Fp::from_canonical(v.to_canonical_u64()).unwrap())
                .collect()
        })
        .collect::<Vec<_>>();
    evaluated.check_columns(&queries, &columns, &answer.proof.siblings)
}

#[test]
fn cpu_batches_use_original_carry_sources_in_both_mode_contexts() {
    let fixtures = json("greedy-boundary/greedy-boundary.json");
    let mut count = 0;
    for (id, c) in fixtures["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["honest"] == true)
        .enumerate()
    {
        let data = Data::read(c);
        let mut masks = || G::ONE;
        let committed_h = pcs::ReferenceCommitment::commit(
            pcs::CommitmentParams {
                log2_k: variables(data.v) as u32,
                log2_inv_rate: 1,
                mask_coeffs: 1,
                row_block: NonZeroUsize::new(1).unwrap(),
                commitment_id: [0x53; 32],
            },
            pcs::RowLayout::new(
                variables(data.n) as u32,
                vec![(0..(1 << variables(data.n))).collect()],
            )
            .unwrap(),
            Secret::new(data.h.clone()),
            &mut masks,
        )
        .unwrap();
        let (cl, g) = context(data.n, data.v);
        let plan = Plan::derive(&g, cl.gadget_policy).unwrap();
        let b = plan
            .greedy_boundary(GadgetId(73), data.prompt as u32)
            .unwrap();
        let pv = fields(&c["target_v"]);
        let po = fields(&c["target_o"]);
        let z = fields(&c["zero"]);
        let target = pv
            .iter()
            .chain(&po)
            .copied()
            .map(independent)
            .collect::<Vec<_>>();
        let tz = z.iter().copied().map(independent).collect::<Vec<_>>();
        let key = extension(&c["key_claim"]);
        let checks = [
            b.key(&target, independent(key)).unwrap(),
            b.zero(Kind::Selected, &tz).unwrap(),
            b.zero(Kind::Tokens, &tz).unwrap(),
        ];
        let inputs =
            [Kind::Key, Kind::Selected, Kind::Tokens].map(|kind| data.build(kind, &pv, &po, &z));
        let instances = inputs.iter().map(Cpu::instance).collect::<Vec<_>>();
        let expected = checks.iter().map(|c| c.instance()).collect::<Vec<_>>();
        for live in [false, true] {
            let public = [vec![key], pv.clone(), po.clone(), z.clone()].concat();
            let (mut p, _) = sessions(live, id as u8, &public, &committed_h.root());
            // The complete model's mixed native batch uses the common degree-seven bound.
            let produced = prove_batch(&mut p, Prefix::Stage(1), MAX_DEGREE, &instances).unwrap();
            let proof = tv::SumcheckProof {
                rounds: produced
                    .proof()
                    .rounds
                    .iter()
                    .map(|r| r.iter().copied().map(independent).collect())
                    .collect(),
                final_evals: produced
                    .proof()
                    .final_evals
                    .iter()
                    .copied()
                    .map(independent)
                    .collect(),
            };
            let verify = |proof: &tv::SumcheckProof| {
                let mut v = sessions(live, id as u8, &public, &committed_h.root()).1;
                let result = tv::verify_batch(
                    &mut v,
                    tv::Prefix::Stage(1),
                    MAX_DEGREE,
                    &expected,
                    proof,
                    |i, point, values| checks[i].evaluate(point, values).ok().map(|f| f.value),
                );
                (result, v)
            };
            let (accepted, mut v) = verify(&proof);
            let accepted = accepted.unwrap();
            assert_eq!(p.state(), *v.state());
            assert_eq!(p.counter(), v.counter());
            let point = accepted.point_of(b.variables()).unwrap();
            let mut at = 0;
            let mut original = Vec::new();
            for check in &checks {
                let end = at + check.instance().evals;
                let bound = check.evaluate(point, &proof.final_evals[at..end]).unwrap();
                assert!(
                    bound
                        .claims
                        .iter()
                        .all(|c| c.value == data.source(c.source, &c.point))
                );
                original.extend(bound.claims);
                at = end;
            }
            assert_eq!(open_onehot(&committed_h, &original, &mut v), Ok(()));
            // Compensating source messages preserve the polynomial, but cannot discharge the
            // ORIGINAL endpoints. Their actual PCS opening must reject the forgery.
            let nt = variables(data.n);
            let nv = variables(data.v);
            let row = &point[nv..];
            let a = Fp2::ONE - row[0];
            let other = if nt == 1 {
                row[0]
            } else {
                row[0] * (Fp2::ONE - row[1])
            };
            assert_ne!(a, Fp2::ZERO);
            assert_ne!(other, Fp2::ZERO);
            let mut changed = proof.clone();
            let start = checks[0].instance().evals;
            changed.final_evals[start + 2] += other;
            changed.final_evals[start + 3] -= a;
            let (forged_sumcheck, mut forged_transcript) = verify(&changed);
            assert!(forged_sumcheck.is_ok());
            let endpoints = checks[1]
                .evaluate(
                    point,
                    &changed.final_evals[start..start + checks[1].instance().evals],
                )
                .unwrap();
            assert_eq!(
                endpoints
                    .claims
                    .iter()
                    .filter(|c| c.value != data.source(c.source, &c.point))
                    .count(),
                2
            );
            let mut forged_sources = Vec::new();
            let mut at = 0;
            for check in &checks {
                let end = at + check.instance().evals;
                forged_sources.extend(
                    check
                        .evaluate(point, &changed.final_evals[at..end])
                        .unwrap()
                        .claims,
                );
                at = end;
            }
            assert_eq!(
                open_onehot(&committed_h, &forged_sources, &mut forged_transcript),
                Err(opening::OpeningError::Evaluation { group: Some(0) })
            );
            for i in 0..proof.final_evals.len() {
                let mut bad = proof.clone();
                bad.final_evals[i] += Fp2::ONE;
                match verify(&bad).0 {
                    Err(tv::SumcheckError::FinalMismatch { round }) => {
                        assert_eq!(round, u32::try_from(b.variables()).unwrap())
                    }
                    Ok(_) => {
                        let mut at = 0;
                        let mut false_sources = 0;
                        for check in &checks {
                            let end = at + check.instance().evals;
                            let f = check.evaluate(point, &bad.final_evals[at..end]).unwrap();
                            false_sources += f
                                .claims
                                .iter()
                                .filter(|c| c.value != data.source(c.source, &c.point))
                                .count();
                            at = end;
                        }
                        assert!(false_sources > 0);
                    }
                    other => panic!("unexpected boundary error {other:?}"),
                }
            }
            count += 1;
        }
    }
    assert_eq!(count, 32);
}
