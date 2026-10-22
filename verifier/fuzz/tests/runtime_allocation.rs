// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Allocation failures in complete runtime metadata stay reader errors (SP-8).
use cap::Cap;
use std::alloc::System;
use tv_core::claims::{ClaimList, QuantPoint};
use tv_core::gadgets::{GadgetError, model, runtime};
use tv_core::validation::{self, ClaimsError};
use tv_testkit::{hex, read_vectors};
#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

fn capped<T, E: core::fmt::Debug + PartialEq>(
    label: &str,
    mut f: impl FnMut() -> Result<T, E>,
    memory: E,
) {
    let before = ALLOCATOR.total_allocated();
    let control = f();
    let bound = ALLOCATOR.total_allocated() - before;
    assert!(control.is_ok());
    drop(control);
    assert!(bound > 0);
    let mut spare = 0;
    let mut refused = 0;
    loop {
        eprintln!("cap {label} spare={spare} bound={bound}");
        ALLOCATOR.set_limit(ALLOCATOR.allocated() + spare).unwrap();
        let result = f().map(drop);
        ALLOCATOR.set_limit(usize::MAX).unwrap();
        if let Err(error) = result {
            assert_eq!(error, memory, "spare={spare}");
            refused += 1;
        } else if spare == bound {
            break;
        }
        assert!(spare < bound, "honest allocation bound must suffice");
        spare = if spare == 0 {
            1
        } else {
            spare.saturating_mul(2).min(bound)
        };
    }
    assert!(refused > 0);
}
#[test]
fn runtime_catalog_layout_leaf_and_mixed_metadata_preserve_memory_errors() {
    capped(
        "range hash",
        || {
            tv_tables::Table::generate(
                tv_tables::TableId(0),
                tv_tables::TableKind::Range { bits: 19 },
            )
        },
        tv_tables::TableError::Memory,
    );
    capped(
        "multi-block hash",
        || {
            tv_tables::Table::generate(
                tv_tables::TableId(0),
                tv_tables::TableKind::Exp2Frac {
                    frac_bits: 14,
                    out_bits: 16,
                },
            )
        },
        tv_tables::TableError::Memory,
    );
    let requant = tv_tables::Table::generate(
        tv_tables::TableId(1),
        tv_tables::TableKind::Requant {
            b: 4,
            r_by_class: vec![2, 3, 2, 3],
            lo: -8,
            hi: 7,
        },
    )
    .unwrap();
    capped(
        "requant regeneration",
        || tv_tables::Table::from_spec(requant.spec()),
        tv_tables::TableError::Memory,
    );
    let range = tv_tables::Table::generate(
        tv_tables::TableId(2),
        tv_tables::TableKind::Range { bits: 19 },
    )
    .unwrap();
    let range_point = vec![tv_field::Fp2::ONE; 19];
    let requant_point = vec![tv_field::Fp2::ONE; 6];
    ALLOCATOR.set_limit(ALLOCATOR.allocated()).unwrap();
    let range_value = range.eval_mle(&range_point, tv_field::Fp2::ONE);
    let requant_value = requant.eval_mle(&requant_point, tv_field::Fp2::ONE);
    ALLOCATOR.set_limit(usize::MAX).unwrap();
    assert!(range_value.is_ok());
    assert!(requant_value.is_ok());
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
    c.arch.d_model = 16;
    c.arch.d_head = 2;
    c.arch.n_heads = 8;
    c.arch.n_kv_heads = 1;
    c.arch.d_ff = 48;
    c.quant.rsqrt_kappa = 0;
    c.quant.exp_sigma = 3;
    c.quant.residual_bits = 8;
    c.quant.residual_exponent = c
        .quant
        .act_exponents
        .iter()
        .find(|(p, _)| *p == QuantPoint::Embed)
        .unwrap()
        .1;
    let public = validation::check_public(&c).unwrap();
    let graph = tv_core::lowering::lower(&public, 5).unwrap();
    let selected = tv_core::selection::derive(&graph, c.gadget_policy).unwrap();
    let tables = selected
        .tables
        .iter()
        .map(|t| tv_tables::Table::generate(t.id, t.kind.clone()).unwrap())
        .collect::<Vec<_>>();
    capped(
        "catalog",
        || runtime::Plan::derive(&graph, c.gadget_policy, 1, 0, &tables),
        ClaimsError::Memory,
    );
    let r = runtime::Plan::derive(&graph, c.gadget_policy, 1, 0, &tables).unwrap();
    assert_eq!(r.catalog().len(), tables.len() + 1);
    let config = tv_core::advice::Config {
        log_k: 6,
        segments: 3,
        query_block: 1,
        heads: 8,
        kv_heads: 1,
        limb_bits: c.gadget_policy.limb_bits,
    };
    capped("advice", || r.advice_layout(config), ClaimsError::Memory);
    capped("leaf", || r.leaf_schedule(), ClaimsError::Memory);
    capped("split", || r.split_sites(), ClaimsError::Memory);
    capped(
        "mixed",
        || model::Plan::derive_runtime(&r),
        GadgetError::Memory,
    );
}
