// SPDX-License-Identifier: Apache-2.0 OR MIT

#![forbid(unsafe_code)]

use tessera_bench::{Counts, MemoryModel, UnitCosts, mu, streaming_passes};

// Exercise the normative implementation with the Python model's exact inputs.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let [
        context,
        batch,
        commit,
        lookup,
        leaf,
        glue,
        weight,
        rate,
        stream,
        answers,
        combine,
        commits,
        lookups,
        scores,
        linear,
        attention,
        regen,
        tables,
        block,
        free,
        bytes,
    ] = args.as_slice()
    else {
        return Err("expected 21 model inputs".into());
    };
    let context = context.parse()?;
    let batch = batch.parse()?;
    let costs = UnitCosts {
        commit_ps: commit.parse()?,
        lookup_ps: lookup.parse()?,
        product_leaf_ps: leaf.parse()?,
        glue_fraction: glue.parse::<f64>()? - 1.0,
        weight_pass_gpu_s: weight.parse()?,
        native_ops_per_s: rate.parse()?,
        streaming_pass_ps: stream.parse()?,
        column_answer_ps: answers.parse()?,
        row_combination_ps: combine.parse()?,
    };
    let counts = Counts {
        commits_per_token: commits.parse()?,
        lookups_per_token: lookups.parse()?,
        scores_per_token_per_ctx: scores.parse()?,
        native_ops_lin: linear.parse()?,
        native_ops_att_per_ctx: attention.parse()?,
        regen_full_equivalents: regen.parse()?,
        table_rows: tables.parse()?,
        score_block: block.parse()?,
    };
    let memory = MemoryModel {
        free_bytes: free.parse()?,
        bytes_per_leaf: bytes.parse()?,
    };
    let m = mu(&counts, &costs, &memory, context, batch)?;
    println!(
        "{} {} {} {} {} {} {} {} {} {}",
        m.linear,
        m.attention,
        m.weights,
        m.regen,
        m.streaming,
        m.column_answers,
        m.row_combination,
        m.tables,
        m.total,
        streaming_passes(&counts, &memory, context, batch)?
    );
    Ok(())
}
