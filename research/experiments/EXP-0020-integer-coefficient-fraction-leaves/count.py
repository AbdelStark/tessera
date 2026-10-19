#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""E0 structural counts independently checked against the Rust actual-loop field trace.

No operation is fitted to an Fp2-equivalent scalar or projected to GPU time. Field inverses,
transcript hashing and allocator instruction costs remain separate opaque operations.
"""
from __future__ import annotations
import json,math,hashlib,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent
B=1024

def buckets(n,k):
    rem=n-k-1;lo=(rem+1)//2;hi=rem//2
    return (1<<hi)*math.ceil((1<<lo)/B)

def structural(L,T,values,arm):
    h=[L.bit_length()-1,T.bit_length()-1]
    terminal=reductions=scaled=0
    for height in h:
        for n in range(height):
            start=2 if arm==2 and n==height-1 else 0
            for k in range(start,n):
                terminal+=4*(1<<(n-k-1)) # 2 evaluated nodes, 2 terminal Fp2 products each
                reductions+=6*buckets(n,k) # two evaluated nodes, three Acc192 components
                scaled+=1<<((n-k)//2) # ceil((n-k-1)/2) low-weight scaling per call
    out={"deferred_wide_products":terminal*4,"unsigned192_additions":terminal*4,"unsigned192_reductions":reductions}
    if arm==0:out={k:0 for k in out}
    signedwide=signedadds=signedreduce=products=adds=extrap=once=conversions=0
    windows=[]
    if arm==2:
        for leaves,table in [(L,False),(T,True)]:
            n=leaves.bit_length()-2;v=2;remaining=1<<(n-v);cells=3**v
            active=(8 if values else 4) if table else(5 if values else 2)
            prods=(8 if values else 3)if table else(4 if values else 1)
            plus=(6 if values else 3)if table else(3 if values else 1)
            bcount=buckets(n,v-1) # rest has n-v variables, unlike a remaining own round
            # buckets(n,v-1) computes the exact split over point[v..].
            signedwide+=remaining*cells*active*4
            signedadds+=remaining*cells*active*2
            signedreduce+=bcount*cells*active*2
            products+=remaining*cells*prods;adds+=remaining*cells*plus
            extrap+=remaining*(2+2*values+2*table)*sum(3**k*2**(v-k-1)for k in range(v))
            once+=leaves*(1+values+table);conversions+=leaves*(1+values+table)
            windows.append({"table":table,"variables":n,"v":v,"cells":cells,"remaining_tuples":remaining,"public_buckets":bcount,"active_channels":active})
    out.update(signed_weight_wide_products=signedwide,signed192_additions=signedadds,signed192_reductions=signedreduce,integer_coefficient_products=products,integer_coefficient_additions=adds,integer_grid_extrapolations=extrap,once_bind_column_reads=once,raw_integer_conversions=conversions)
    return out,windows,scaled

def logical_passes(L,T,values,arm):
    """Logical element accesses, not cache transactions/compiler load instructions.

    Ordinary folds read each input pair once and write one result. Raw once-bind
    reads every original coordinate once, then ordinary folds resume. Kernel node
    scan reads are classified separately because eq weights/denominators are
    re-read at every own round (even during C windows).
    """
    fold_reads=fold_writes=raw_reads=raw_writes=eq_entries=grid_reads=0
    window_acc_clear=window_grid_clear=window_stack_clear=deferred_stack_clear=0
    scan_d=scan_n=raw_window=weighted_eq_reads=0
    rows=[]
    for leaves,table in [(L,False),(T,True)]:
        h=leaves.bit_length()-1
        for n in range(1,h):
            bottom=n==h-1;fields=2 if bottom and not table else 4
            # add_round builds both split eq tables even when Window supplies node_sums.
            eq_entries+=sum((1<<((n-k)//2))+(1<<((n-k-1)//2))for k in range(n))
            if arm==2 and bottom:
                v=2;remaining=1<<(n-v)
                raw_reads+=leaves*(1+values+table)
                raw_window+=leaves*(1+values+table)
                raw_writes+=fields*remaining
                fw=fields*(remaining-1)
                eq_entries+=(1<<((n-v+1)//2))+(1<<((n-v)//2))+4+2+1
                grid_reads+=sum(3*(1<<(v-k-1))for k in range(v))
                window_acc_clear+=buckets(n,v-1)*9*8*2*24
                window_grid_clear+=9*8*16+(9+3+1)*16+2*16
                window_stack_clear+=6*8*8+6*27*8
                start=v
            else:
                fw=fields*((1<<n)-1);start=0
            for k in range(start,n):
                tuples=1<<(n-k-1)
                # P fused nodes0/2 share loaded rows; D evaluates two terminal nodes.
                scan_d+=(4 if arm==0 else 8)*tuples
                if not(bottom and not table):scan_n+=(4 if arm==0 else 8)*tuples
                weighted_eq_reads+=tuples
            fold_writes+=fw;fold_reads+=2*fw
            if arm:
                deferred_stack_clear+=sum(216*buckets(n,k)for k in range(start,n))
            rows.append({'table':table,'variables':n,'ordinary_fold_input_reads':2*fw,'ordinary_fold_output_writes':fw,'raw_once_bind':arm==2 and bottom})
    return {'node_scan_denominator_element_reads':scan_d,'node_scan_numerator_element_reads':scan_n,'node_scan_eqlo_element_reads':weighted_eq_reads,'C_window_raw_coordinate_reads':raw_window,'ordinary_fold_input_element_reads':fold_reads,'ordinary_fold_output_element_writes':fold_writes,'raw_once_bind_coordinate_reads':raw_reads,'raw_once_bind_output_writes':raw_writes,'split_and_window_eq_entries_constructed':eq_entries,'C_window_contraction_grid_reads':grid_reads,'C_window_accumulator_allocation_bytes_cleared':window_acc_clear,'C_window_sums_grid_challenge_capacity_bytes_cleared':window_grid_clear,'C_window_explicit_integer_stack_bytes_cleared':window_stack_clear,'D_explicit_stack_accumulator_bytes_cleared':deferred_stack_clear,'per_level_folds':rows,'scope':'Node scan counts name source-level accesses before compiler reuse; P fuses nodes0/2, D visits each node. Logical principal table/grid/fold passes and explicit clearing, not physical memory transactions. Eq Vecs are public and not zeroized; field arithmetic includes all node scans, eq products, interpolation and folds in the actual-loop trace.'}

def buffers(L,T,values,arm):
    # Exact field-buffer capacities at completed tree construction, excluding Vec/enum metadata.
    inner=32*(L+T-4)
    raw=(8*(1+values)*L+8*(2+values)*T) if arm==2 else 16*L+24*T
    negated=8*T
    root_fields=inner+raw+negated
    bound_clear=0;scaled_clear=0
    for leaves,table in [(L,False),(T,True)]:
        h=leaves.bit_length()-1
        for n in range(1,h):
            bottom=n==h-1
            if arm==2 and bottom:
                bound_clear+=4*16*(leaves//8) # all four buffers reserve, including unit N0/N1
                start=2
            else:
                fields=2 if bottom and not table else 4
                bound_clear+=fields*16*(1<<(n-1));start=0
            if arm:
                for k in range(start,n):scaled_clear+=16*(1<<((n-k)//2))
    source_envelope=(40*L+56*T) if arm==2 and not values else 48*L+64*T
    peak_cap=4*source_envelope+64*1024**2
    assert peak_cap<=4*1024**3
    return {"constructed_field_buffer_bytes_with_negated_m":root_fields,"internal_level_field_bytes":inner,"leaf_field_or_integer_bytes":raw,"negated_m_bytes":negated,"bound_buffer_capacity_bytes_cleared":bound_clear,"scaled_weight_capacity_bytes_cleared":scaled_clear,"C_unused_unit_numerator_capacity_bytes":(4*L if arm==2 else 0),"conservative_total_scratch_upper_bytes":peak_cap,"limit_bytes":4*1024**3,"caveat":"Exact principal field-buffer capacities plus a conservative total ceiling; not allocator telemetry or a measured RSS peak. Proof/event/Vec metadata and transient scalar grids fit the separate ceiling margin."}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=HERE/'results');results=parser.parse_args().results
    trace=json.loads((results/'count-trace.json').read_text())
    preflight=json.loads((results/'preflight.json').read_text())
    cases={c['case']:c for c in preflight['cases']};assert len(cases)==6 and len(trace['records'])==18
    records=[]
    for row in trace['records']:
        c=cases[row['case']];arm=row['arm'];L=c['L'];T=c['T'];values=c['profile']!='R'
        expected,windows,scaled=structural(L,T,values,arm)
        ops=row['operations'];assert bytes(row['statement_state']).hex()==c['production']['state']
        for k,v in expected.items():assert ops[k]==v,(row['case'],arm,k,ops[k],v)
        mem=buffers(L,T,values,arm);assert mem['conservative_total_scratch_upper_bytes']==c['arms'][arm]['scratch_upper_bytes']
        derived={"unsigned192_base_mul_const":2*ops['unsigned192_reductions'],"unsigned192_base_add":ops['unsigned192_reductions'],"unsigned192_base_sub":ops['unsigned192_reductions'],"signed192_base_mul_const":2*ops['signed192_reductions'],"signed192_base_add":ops['signed192_reductions'],"signed192_base_sub_including_sign_correction":2*ops['signed192_reductions'],"schoolbook_result_mul7":ops['unsigned192_reductions']//3,"schoolbook_result_add":ops['unsigned192_reductions']//3,"deferred_192_word_adds":5*ops['unsigned192_additions'],"signed_192_word_adds":9*ops['signed192_additions'],"signed_weight_u128_adds":2*ops['signed192_additions'],"signed_weight_magnitude_u128_xor_and_sub_each":ops['signed192_additions'],"signed_weight_sign_u64_xors":3*ops['signed192_additions'],"signed_weight_sign_mask_u64_subtractions":ops['signed192_additions'],"base_multiplicity_negations":T,"C_root_cross_extra_base_multiplicity_negations":T if arm==2 else 0,"ordinary_bottom_table_z2_base_additions":T//2 if arm!=2 else 0,"ordinary_bottom_table_z2_base_subtractions":T//2 if arm!=2 else 0,"ordinary_bottom_table_first_bind_base_subtractions":T//2 if arm!=2 else 0,"C_root_integer_to_field_conversions":ops['raw_integer_conversions'],"C_once_bind_integer_to_field_conversions":ops['once_bind_column_reads']}
        H=max(L,T).bit_length()-1
        records.append({"case":c['case'],"arm":arm,"original_leaves":L+T,"semantic_operations":ops,"operations_per_original_leaf":{k:v/(L+T)for k,v in ops.items()},"derived_explicit_base_word_operations":derived,"memory":mem,"logical_passes_and_temporary_clearing":logical_passes(L,T,values,arm),"windows":windows,"deferred_scaled_weights_constructed":scaled if arm else 0,"transcript_calls":{"Fp2_challenges_opaque_internal_rejection":2+3*H+H*(H-1)//2,"LogUp_absorbs":2+H+H*(H-1)//2,"case_descriptor_absorbs":1,"prefix_events_paid_by_diagnostic":H*H+H+1},"scope":"whole synthetic unmasked resident LogUp proof; no GPU or complete-LM rescore","opaque_shared_scalar_control":"Base half inverse, padding powers and integer-to-field canonical conversion internals are public/small opaque controls; extension inverses/pow, hashes and allocator instructions are not expanded. Operator census is semantic, not a full scalar instruction counter."})
    out={"evidence":"E0 semantic exact-loop trace and independently derived structural counts; no field inverse/hash/allocator cost fit","primary_v":2,"records":records}
    (results/'count.json').write_text(json.dumps(out,indent=2)+'\n');print('count pass:',len(records),'records')
if __name__=='__main__':main()
