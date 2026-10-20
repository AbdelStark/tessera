// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Proof-controlled counts are rejected before any gadget-owned allocation (SP-8).
use cap::Cap;
use std::alloc::System;
use tv_core::claims::{ClaimList, QuantPoint};
use tv_core::gadgets::{
    GadgetError, Plan,
    argmax::{Argmax, Claim, Role},
    argmax_outputs::Binding,
    comparison::{Comparison, Values as CompareValues},
    decomposition::{Decomposition, Values},
    root_products::Products,
};
use tv_core::lowering;
use tv_core::validation::{self, GadgetId};
use tv_field::Fp2;
use tv_testkit::{hex, read_vectors};
#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);
fn measured<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATOR.total_allocated();
    let result = f();
    (result, ALLOCATOR.total_allocated() - before)
}
fn base() -> ClaimList {
    let file = read_vectors("claims/claim-lists.json");
    let case = file
        .get("cases")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c.get("model").unwrap().as_str() == Some("tiny-2-servable.d"))
        .unwrap();
    let mut c = ClaimList::open(
        &hex(case.get("hex").unwrap().as_str().unwrap()),
        &tv_codec::DecodeLimits::MESSAGE,
    )
    .unwrap();
    c.arch.d_model = 2;
    c.arch.d_head = 2;
    c.arch.n_heads = 1;
    c.arch.n_kv_heads = 1;
    c.arch.d_ff = 6;
    c.arch.vocab = 5;
    c.arch.max_context = 8;
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
    c
}
#[test]
fn million_element_gadget_inputs_reject_without_proof_sized_allocations() {
    let c = base();
    let public = validation::check_public(&c).unwrap();
    let graph = lowering::lower(&public, 3).unwrap();
    let plan = Plan::derive(&graph, c.gadget_policy).unwrap();
    let huge = vec![Fp2::ZERO; 1 << 20];
    let gid = GadgetId(73);
    let d = Decomposition::derive(55, 16).unwrap();
    let (result, bytes) = measured(|| {
        d.check(
            gid,
            Values {
                point: &[],
                value: Fp2::ZERO,
                low: &huge,
                keys: &huge,
            },
        )
        .map(|c| c.claims().count())
    });
    assert!(matches!(result,Err(GadgetError::Rejected{gadget})if gadget==gid));
    assert_eq!(bytes, 0);
    let cmp = Comparison::derive(8, &[255, 0], &[255, 0]).unwrap();
    let (result, bytes) = measured(|| {
        cmp.check(
            gid,
            CompareValues {
                point: &[],
                left: &huge,
                right: &huge,
                arrays: [&huge; 6],
            },
        )
        .map(|c| c.claims().count())
    });
    assert!(matches!(result,Err(GadgetError::Rejected{gadget})if gadget==gid));
    assert_eq!(bytes, 0);
    let product = Products::derive(32).unwrap();
    let (result, bytes) = measured(|| product.evaluate(0, false, &huge));
    assert_eq!(result, None);
    assert_eq!(bytes, 0);
    let root = plan.roots().next().unwrap().unwrap();
    let point = vec![Fp2::ONE; root.padding_instance().vars];
    let sources = vec![Fp2::ZERO; root.sources().len()];
    let keys = vec![Fp2::ZERO; root.tables().len()];
    let values = keys.clone();
    let (honest, bytes) = measured(|| {
        root.check(&point, &sources, &keys, &values)
            .map(|c| c.claims().count())
    });
    assert!(honest.is_ok());
    assert_eq!(bytes, 0);
    for (s, k, v) in [
        (&huge[..], &keys[..], &values[..]),
        (&sources[..], &huge[..], &huge[..]),
        (&sources[..], &keys[..], &huge[..]),
    ] {
        let (result, bytes) = measured(|| root.check(&point, s, k, v).map(|c| c.claims().count()));
        assert!(matches!(result,Err(GadgetError::Rejected{gadget})if gadget==root.gadget()));
        assert_eq!(bytes, 0);
    }
    let softmax = plan.softmaxes().next().unwrap().unwrap();
    let point = vec![Fp2::ONE; softmax.row_variables()];
    let count = (0..128)
        .take_while(|&i| softmax.source(i).is_some())
        .count();
    let sources = vec![Fp2::ZERO; count];
    let keys = vec![Fp2::ZERO; softmax.range_tables().len()];
    let values = keys.clone();
    let (honest, bytes) = measured(|| {
        softmax
            .check(&point, &sources, &keys, &values)
            .map(|c| c.claims().count())
    });
    assert!(honest.is_ok());
    assert!(bytes <= 35 * 16);
    for (s, k, v) in [
        (&huge[..], &keys[..], &values[..]),
        (&sources[..], &huge[..], &huge[..]),
        (&sources[..], &keys[..], &huge[..]),
    ] {
        let (result, bytes) =
            measured(|| softmax.check(&point, s, k, v).map(|c| c.claims().count()));
        assert!(matches!(result,Err(GadgetError::Rejected{gadget})if gadget==softmax.gadget()));
        assert_eq!(bytes, 0);
    }
    let requant = plan.requantizations().next().unwrap().unwrap();
    let node = graph.node(requant.node()).unwrap();
    let dims = graph
        .tensor(*node.outputs().first().unwrap())
        .unwrap()
        .shape
        .dims();
    let vars = dims
        .iter()
        .map(|&n| (u32::BITS - (n - 1).leading_zeros()) as usize)
        .sum();
    let point = vec![Fp2::ONE; vars];
    let tuples = vec![(Fp2::ZERO, Fp2::ZERO); 1 << 20];
    let (result, bytes) = measured(|| {
        requant
            .check(&point, &huge, &tuples)
            .map(|c| c.claims().count())
    });
    assert!(matches!(result,Err(GadgetError::Rejected{gadget})if gadget==requant.gadget()));
    assert_eq!(bytes, 0);
    let onehot = plan.one_hots().next().unwrap().unwrap();
    let row = [Fp2::ONE; 2];
    let at = onehot.at(&row, Fp2::ZERO).unwrap();
    let point = [Fp2::ZERO; 5];
    let (result, bytes) = measured(|| {
        at.evaluate(
            tv_core::gadgets::onehot::Relation::Booleanity,
            &point,
            &point,
            &huge,
        )
        .map(|c| c.claims().count())
    });
    assert!(matches!(result,Err(GadgetError::Rejected{gadget})if gadget==onehot.gadget()));
    assert_eq!(bytes, 0);
    let argmax = Argmax::derive(gid, 5, 2).unwrap();
    let point = [Fp2::ZERO; 4];
    let key = argmax.lookup(&point, Fp2::ZERO, Fp2::ZERO).unwrap();
    let (result, bytes) = measured(|| key.evaluate(&point, &huge).map(|c| c.claims().count()));
    assert!(matches!(result,Err(GadgetError::Rejected{gadget})if gadget==gid));
    assert_eq!(bytes, 0);
    let outputs = plan.argmax_outputs(gid, 1).unwrap();
    let Binding::Graph(view) = outputs
        .bind(Claim {
            role: Role::Indicator,
            point: &point,
            value: Fp2::ZERO,
        })
        .unwrap()
    else {
        panic!("graph view")
    };
    let source_point = [Fp2::ZERO; 2];
    let (result, bytes) = measured(|| view.evaluate(&source_point, &huge).map(|c| c.value()));
    assert!(matches!(result,Err(GadgetError::Rejected{gadget})if gadget==gid));
    assert_eq!(bytes, 0);
}
