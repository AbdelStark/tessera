#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""EXP-0016: exact finite certificates and complete merged read identities.

This is a relation model, not a verifier implementation or a probabilistic proof.
Tuple Counters test the global identity before challenge compression; explicit fraction
evaluations independently check the attacks. Every input is synthetic.
"""

from __future__ import annotations

from collections import Counter
from itertools import product, permutations
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
F = 65537
GOLD = 2**64 - 2**32 + 1
SIGMA = 4
BASE_ROWS = 64
DOUBLED_ROWS = 128

# Reuse the normative independent integer generator, without running its vector writer.
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("exp16_table_oracle", ROOT / "spec/vectors/tables/generate.py")
assert spec is not None and spec.loader is not None
table_oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table_oracle)
BASE_EXP = [table_oracle.exp_value(k, SIGMA, 1, 1) for k in range(BASE_ROWS)]
assert BASE_EXP[:4] == [16, 6, 2, 1] and BASE_EXP[-1] == 0
EXP_CEIL = [BASE_EXP[(k + 1)//2] if (k + 1)//2 < BASE_ROWS else 0 for k in range(DOUBLED_ROWS)]
EXP_FLOOR = [BASE_EXP[k//2] for k in range(DOUBLED_ROWS)]


def chunks(j: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((int(j//2 == 0), int(j//2 == 1)), (int(j % 2 == 0), int(j % 2 == 1)))


def lt_from_chunks(i: int, oh: tuple[tuple[int, int], tuple[int, int]]) -> int:
    high, low = i//2, i % 2
    return sum(v * (high < k) for k, v in enumerate(oh[0])) + oh[0][high] * sum(v * (low < k) for k, v in enumerate(oh[1]))


def oracle(scores: tuple[int, ...], q: int) -> tuple[int, int]:
    maximum = max(scores[:q+1])
    return maximum, next(i for i in range(q+1) if scores[i] == maximum)


def max_relations(scores: tuple[int, ...], q: int, j: int, c: int, *, causal: bool = True, parity: bool = True, floor: bool = False, modulus: int = F, base_rows: int = BASE_ROWS) -> dict | None:
    """Evaluate every padded score position; c is supplied by the read/dot equation."""
    if causal and j > q:
        return None
    oh = chunks(j)
    keys, values, comparisons = [], [], []
    for i in range(4):
        lt = lt_from_chunks(i, oh)
        assert lt == int(i < j)
        comparisons.append(lt)
        key = (2 * (c - scores[i]) - (lt if parity else 0)) % modulus if i <= q and i < len(scores) else 2*base_rows - 1
        if key >= 2*base_rows:
            return None
        keys.append(key)
        index = key//2 if floor else (key+1)//2
        values.append(0 if index == base_rows else BASE_EXP[index] if index < BASE_ROWS else table_oracle.exp_value(index,SIGMA,1,1))
    return {"j": j, "c_integer": c, "c_field": c % modulus, "chunks": oh, "lt": comparisons, "keys": keys, "e": values}


def bracket(values: list[int], psi: int = 1) -> dict:
    total = sum(values)
    reciprocal = (1 << (2 * SIGMA)) // total if total else 0
    a = psi * (1 << (2*SIGMA)) - reciprocal * total
    b = total - psi - a
    okay = (0 <= reciprocal < 1 << (SIGMA+1) and 0 <= a < 1 << (SIGMA+18) and 0 <= b < 1 << (SIGMA+18))
    if not psi:
        okay &= reciprocal == 0
    return {"sum": total, "R": reciprocal, "a": a, "b": b, "a_limbs16": [a & 65535, a >> 16], "b_limbs16": [b & 65535, b >> 16], "passes": bool(okay)}


def factor_exhaustions() -> dict:
    roots = [x for x in range(F) if x * (x-1) % F == 0]
    assert roots == [0, 1]
    booleans = list(product(roots, repeat=4))
    onehot = [v for v in booleans if sum(v[:2]) % F == 1 and sum(v[2:]) % F == 1]
    assert len(onehot) == 4
    for i, j in product(range(4), repeat=2):
        expected = lt_from_chunks(i, chunks(j))
        got = [x for x in range(F) if (x - expected) % F == 0]
        assert got == [int(i < j)]
    for expected in range(-8, 9):
        c_roots = [x for x in range(F) if (x - expected) % F == 0]
        read_roots = [x for x in range(F) if x == expected % F]
        assert c_roots == read_roots == [expected % F]
    values = sorted(set(EXP_CEIL))
    for value in values:
        roots_out = [x for x in range(F) if (x-value) % F == 0]
        assert roots_out == [value]
    all_keys = []
    for key in range(DOUBLED_ROWS):
        e = EXP_CEIL[key]
        assert 0 <= e <= 16
        all_keys.append({"key": key, "ceil": e, "floor": EXP_FLOOR[key]})
    return {"field": F, "boolean_candidates": F, "boolean_roots": roots, "boolean_chunk_assignments": len(booleans), "sum_one_assignments": len(onehot), "lt_candidates": 16*F, "c_candidates": 17*F, "Kread_candidates": 17*F, "output_candidates": len(values)*F, "exp2_rows": all_keys, "top_row_requires_extension": True}


def local_exhaustion() -> dict:
    result = []
    for length in (3, 4):
        rows = prefixes = candidates = accepted = 0
        for scores in product(range(-8, 9), repeat=length):
            rows += 1
            for q in range(length):
                prefixes += 1
                expected_c, expected_j = oracle(scores, q)
                witnesses = []
                for j in range(4):
                    candidates += 1
                    read = scores[j] if j < length else 0
                    witness = max_relations(scores, q, j, read)
                    if witness is None:
                        continue
                    assert j == expected_j and read == expected_c
                    assert witness["e"] == [BASE_EXP[expected_c-scores[i]] if i <= q else 0 for i in range(4)]
                    witnesses.append(witness)
                assert len(witnesses) == 1, (scores, q, witnesses)
                accepted += len(witnesses)
        result.append({"length": length, "rows": rows, "prefixes": prefixes, "candidate_j": candidates, "unique_witnesses": accepted})
    # Independent actual dot-product inputs, rather than arbitrary score rows alone.
    dot_cases = 0
    vectors = list(product((-1, 0, 1), repeat=2))
    for query, left, right in product(vectors, repeat=3):
        scores = (sum(q*k for q,k in zip(query,left)), sum(q*k for q,k in zip(query,right)))
        for q in range(2):
            passing = [j for j in range(2) if max_relations(scores, q, j, scores[j]) is not None]
            assert passing == [oracle(scores,q)[1]]
            dot_cases += 1
    return {"row_domains": result, "d2_dot_prefixes": dot_cases, "field_checks_use_read_forced_c": True}


def adversarial_controls() -> dict:
    tied = (0,0,0,0)
    assert max_relations(tied,3,2,0) is None
    later = max_relations(tied,3,2,0,parity=False)
    assert later is not None
    future = max_relations((0,1,0),0,1,1,causal=False)
    assert future is not None and max_relations((0,1,0),0,1,1) is None
    correct = max_relations((0,1,0),1,1,1)
    shifted = max_relations((0,1,0),1,1,1,floor=True)
    assert correct is not None and shifted is not None
    assert correct["e"][0] == 6 and shifted["e"][0] == 16
    return {"tie_without_parity": later, "future_without_causality": future, "ceil_variant": correct, "literal_shift_variant": shifted, "odd_key_semantics_differ": True}


def address(layer: int, head: int, position: int, dim: int, mode: str, *, heads: int = 2, positions: int = 2, dimensions: int = 2) -> int:
    l = layer if mode not in ("untagged", "omit-layer") else 0
    h = head if mode not in ("untagged", "omit-head") else 0
    if mode == "omit-dimension":
        return (l*heads+h)*positions+position
    return (((l*heads+h)*positions+position)*dimensions+dim)


def counter_json(c: Counter) -> list[dict]:
    return [{"tuple": list(k), "count": v} for k,v in sorted(c.items()) if v]


def fractions(left: Counter, right: Counter, modulus: int) -> list[dict]:
    records = []
    for alpha,beta in ((1009,17),(10007,31)):
        def total(entries: Counter) -> int:
            out = 0
            for (key,value), multiplicity in entries.items():
                denominator = (alpha-key-beta*value) % modulus
                assert denominator != 0
                out = (out + multiplicity * pow(denominator,-1,modulus)) % modulus
            return out
        l,r = total(left), total(right)
        records.append({"alpha":alpha,"beta":beta,"lookup_sum":l,"table_sum":r,"residual":(l-r)%modulus})
    return records


def scalar_merged_exhaustion() -> dict:
    order = [(l,h) for l in range(2) for h in range(2)]
    total = sublemma_false = full_bracket_false = routed_false = alternatives = 0
    first = None
    for table in product((-1,0,1), repeat=4):
        table_counter = Counter((0,x) for x in table)
        for queries in product((-1,0,1), repeat=4):
            for perm in permutations(range(4)):
                total += 1
                reads = [table[i] for i in perm]
                assert Counter((0,x) for x in reads) == table_counter
                certificates = [max_relations((queries[i]*table[i],),0,0,queries[i]*reads[i]) for i in range(4)]
                if any(c is None for c in certificates):
                    continue
                wrong_c = any(queries[i]*reads[i] != queries[i]*table[i] for i in range(4))
                wrong_read = reads != list(table)
                if wrong_read:
                    alternatives += 1
                if wrong_c:
                    sublemma_false += 1
                if wrong_read and all(bracket(c["e"])["passes"] for c in certificates):
                    full_bracket_false += 1
                tagged_table = Counter((address(*order[i],0,0,"full",positions=1,dimensions=1),table[i]) for i in range(4))
                tagged_reads = Counter((address(*order[i],0,0,"full",positions=1,dimensions=1),reads[i]) for i in range(4))
                if tagged_reads == tagged_table:
                    assert not wrong_read and not wrong_c
                    routed_false += int(wrong_read or wrong_c)
                if wrong_read and first is None:
                    first = {"table":table,"queries":queries,"permutation":perm,"reads":reads,"certificates":certificates,"m":[1]*4,"global_counter":counter_json(table_counter)}
    assert sublemma_false and full_bracket_false and routed_false == 0
    return {"complete_configs": total, "wrong_max_sublemma_certificates":sublemma_false,"alternative_read_certificates":alternatives,"alternative_reads_with_reciprocal_brackets":full_bracket_false,"fully_tagged_false_certificates":routed_false,"first_alternative":first}


def merged_attack(kind: str, *, positions: int = 2, dimensions: int = 2, modulus: int = F, base_rows: int = BASE_ROWS) -> dict:
    layers = heads = group = 2
    table = {}
    for l,h,i in product(range(layers), range(heads), range(positions)):
        if kind == "dimension":
            key = (0,1+i)
        else:
            cls = h if kind == "head" else l
            key = (cls,-i)
        table[l,h,i] = key + (0,)*(dimensions-2)
    reads = []
    m = Counter()
    wrong_rows = []
    for l,h,qhead,q in product(range(layers),range(heads),range(group),range(positions)):
        global_head = h*group+qhead
        assert global_head//group == h
        cls = h if kind == "head" else l
        query = ((1 if cls == 0 else -1),0) if qhead == 0 else (0,1)
        if kind == "dimension":
            query = (1,0) if qhead == 0 else (0,1)
        query += (0,)*(dimensions-2)
        scores = tuple(sum(a*b for a,b in zip(query,table[l,h,i])) for i in range(positions))
        honest_c,honest_j = oracle(scores,q)
        j = honest_j
        source_l,source_h = l,h
        cheating = q == 1 and qhead == 0
        if cheating and kind in ("layer","both"):
            source_l = 1-l
        if cheating and kind in ("head","both"):
            source_h = 1-h
        kread = table[source_l,source_h,j]
        if cheating and kind == "dimension":
            kread = tuple(reversed(kread))
        c = sum(a*b for a,b in zip(query,kread))
        certificate = max_relations(scores,q,j,c,modulus=modulus,base_rows=base_rows)
        assert certificate is not None
        rb = bracket(certificate["e"])
        assert rb["passes"], (kind,l,h,qhead,q,c,rb)
        product_root = 1
        for i in range(q+1):
            product_root = product_root * (c-scores[i]) % modulus
        if cheating:
            assert c > honest_c
            assert product_root != 0
            wrong_rows.append([l,global_head,q])
        else:
            assert c == honest_c
            assert product_root == 0
        # A single multiplicity per position supplies both dimension columns.
        m[source_l,source_h,j] += 1
        reads.append({"layer":l,"query_head":global_head,"kv_head":h,"q":q,"query":query,"scores":scores,"j":j,"source_layer":source_l,"source_kv_head":source_h,"Kread":kread,"honest_c":honest_c,"honest_j":honest_j,"certificate":certificate,"reciprocal":rb,"current_product_root":product_root,"current_product_certificate_accepts":product_root == 0,"wrong":cheating})
    variants = {}
    for mode in ("untagged","omit-layer","omit-head","omit-dimension","full"):
        lookup = Counter()
        weighted_table = Counter()
        for r in reads:
            for dim,value in enumerate(r["Kread"]):
                lookup[address(r["layer"],r["kv_head"],r["j"],dim,mode,positions=positions,dimensions=dimensions),value] += 1
        for (l,h,i), values in table.items():
            for dim,value in enumerate(values):
                weighted_table[address(l,h,i,dim,mode,positions=positions,dimensions=dimensions),value] += m[l,h,i]
        support = {(address(l,h,i,dim,mode,positions=positions,dimensions=dimensions),value) for (l,h,i),values in table.items() for dim,value in enumerate(values)}
        missing = sorted(set(lookup)-support)
        equal = lookup == weighted_table
        diag = fractions(lookup,weighted_table,F)
        gold = fractions(lookup,weighted_table,GOLD)
        if equal:
            assert all(x["residual"] == 0 for x in diag+gold)
        variants[mode] = {"full_tuple_identity":equal,"lookup_tuples_absent_from_entire_table":missing,"lookup_multiset":counter_json(lookup),"weighted_table_multiset":counter_json(weighted_table),"F65537_fractions":diag,"Goldilocks_fractions":gold}
    primary = "omit-dimension" if kind == "dimension" else "untagged"
    assert variants[primary]["full_tuple_identity"] and not variants["full"]["full_tuple_identity"]
    assert variants["full"]["lookup_tuples_absent_from_entire_table"], "changing m cannot fix an absent routed tuple"
    assert wrong_rows
    return {"attack":kind,"dimensions":{"layers":layers,"kv_heads":heads,"query_heads_per_kv":group,"positions":positions,"d":dimensions},"K":[{"layer":l,"kv_head":h,"i":i,"value":v}for (l,h,i),v in sorted(table.items())],"m":[{"layer":l,"kv_head":h,"i":i,"count":m[l,h,i]}for l,h,i in sorted(table)],"rows":reads,"wrong_rows":wrong_rows,"variants":variants,"every_reciprocal_bracket_passes":True,"scope":"relation counterexample, not a forged production proof"}


def padded_read_accepts(value: int, psi: int) -> bool:
    return psi != 0 or value % F == 0


def padding_checks() -> dict:
    zero_chunks = (0,0,0,0)
    literal_sum_one = sum(zero_chunks[:2]) == 1 and sum(zero_chunks[2:]) == 1
    assert not literal_sum_one
    padded_boolean = [v for v in product((0,1),repeat=4) if sum(v[:2]) % F == 0 and sum(v[2:]) % F == 0]
    assert padded_boolean == [zero_chunks]
    naive_reads = list(range(F))  # ψ-gated read and Q=0 impose no relation on this coordinate.
    pinned_reads = [x for x in range(F) if padded_read_accepts(x,0)]
    assert len(naive_reads) == F and pinned_reads == [0]
    pad = max_relations((0,0,0),2,0,0)
    assert pad is not None and pad["e"][3] == 0
    assert bracket([0,0,0,0],psi=0)["passes"]
    selector_rows = []
    for layer,head,q in product(range(4),repeat=3):
        # Three real layers, three KV/query heads (GQA group1), n=3; all axes pad to4.
        psi = int(layer < 3 and head < 3 and q < 3)
        oh = chunks(0) if psi else ((0,0),(0,0))
        assert all(sum(chunk) == psi for chunk in oh)
        kread = (0,0)
        c = psi * sum(a*b for a,b in zip((0,0),kread))
        assert c == 0 and all((1-psi)*value == 0 for value in kread)
        keys,values,selectors = [],[],[]
        for i in range(4):
            causal = int(layer < 3 and head < 3 and i <= q < 3)
            lt = lt_from_chunks(i,oh)
            key = causal * (2*c-lt) + (1-causal)*127
            assert 0 <= key < 128
            keys.append(key);values.append(EXP_CEIL[key]);selectors.append(causal)
        rb = bracket(values,psi)
        assert rb["passes"]
        if not psi:
            assert keys == [127]*4 and values == [0]*4 and rb["R"] == 0
            # A gated read sees no lookup, so a nonzero padded Kread is a second naive witness.
            alternative_kread = (1,0)
            assert psi * sum(a*b for a,b in zip((0,0),alternative_kread)) == c
            assert any((1-psi)*x for x in alternative_kread)
        selector_rows.append({"layer":layer,"kv_head":head,"query_head":head,"q":q,"psi":psi,"C":selectors,"chunks":oh,"keys":keys,"e":values,"c":c,"Kread":kread,"reciprocal":rb})
    assert sum(r["psi"] for r in selector_rows) == 27
    assert len(selector_rows) == 64
    return {"literal_zero_padded_chunks_complete":literal_sum_one,"literal_failure":"sum-one chunks conflict with all-zero public padding","naive_read_gated_Kread_count_per_coordinate":len(naive_reads),"repaired_Kread_count_per_coordinate":len(pinned_reads),"repaired_chunks":padded_boolean,"selector_rows":selector_rows,"padded_types":["q>=n","layer>=n_layers","query_head>=n_heads","KV-head table padding"],"length3_padded_to4_masked_e":pad["e"][3],"repaired_c":0,"repaired_R":0,"literal_real_row_all_zero_witness":max_relations((0,0,0,0),3,0,0)}


def goldilocks_obligations() -> dict:
    def unpack(index: int, heads: int, positions: int, dim: int) -> tuple[int,int,int,int]:
        index,d = divmod(index,dim);index,i = divmod(index,positions);l,h = divmod(index,heads)
        return l,h,i,d
    normal = (126,8,1<<17,128)
    huge = (65535,65535,1<<17,65534)
    normal_total = normal[0]*normal[1]*normal[2]*normal[3]
    huge_total = huge[0]*huge[1]*huge[2]*huge[3]
    assert normal_total < GOLD < huge_total
    alias = unpack(GOLD,*huge[1:])
    assert all(0 <= x < limit for x,limit in zip(alias,huge))
    packed = (((alias[0]*huge[1]+alias[1])*huge[2]+alias[2])*huge[3]+alias[3])
    assert packed == GOLD and packed % GOLD == 0
    bounds = []
    for d in (128,65534):
        low = -128*127*d
        high = 128*128*d
        spread = 128*255*d
        assert high-low == spread and 2*spread+1 < (GOLD-1)//2
        bounds.append({"d":d,"score_min":low,"score_max":high,"spread":spread,"doubled_key_abs_bound":2*spread+1,"below_centered_Goldilocks_half":True})
    assert bounds[0]["score_max"] == 1<<21
    return {"P":GOLD,"bounds":bounds,"onehot_requirement":"chunk_length<P; sum-one then means exactly one Boolean one","compressed_P_ones_padding_alias":{"ones":GOLD,"sum_modP":0},"compressed_P_plus_1_ones_sum_one_alias":{"ones":GOLD+1,"sum_modP":1},"405B":{"shape":normal,"namespace_size":normal_total,"injective":True},"syntactic_large_shape":{"shape":huge,"namespace_size":huge_total,"d_model":huge[1]*huge[3],"full_statement_and_resource_acceptance_not_checked":True,"aliases":[[0,0,0,0],alias],"packed_residues":[0,packed%GOLD]},"general_theorem_requirements":["injective layer/KV/position/dimension namespace or separate instance identities","table values tied to the mandated committed post-Requant tensor","true Boolean chunk count smaller than field characteristic","complete functional ceil-based Exp2 with public zero-tail masked key","C and psi gating plus unique padded Kread/chunks/c/R","commit all values and m before random compression","separate ordinary statistical LogUp/zerocheck bounds","opening, masking and transcript integration remain unimplemented"]}


def main() -> None:
    frozen = (HERE/"results/README.preregistered.txt").read_bytes()
    committed = subprocess.check_output(["git","show","dab7dc1:research/experiments/EXP-0016-argmax-read-lemma/README.md"],cwd=ROOT)
    assert frozen == committed
    result = {"experiment":"EXP-0016","evidence":"E0","date_utc":datetime.now(timezone.utc).isoformat(),"environment":{"python":sys.version,"platform":platform.platform(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()},"preregistration_sha256":hashlib.sha256(frozen).hexdigest()}
    result["field_factors"] = factor_exhaustions()
    print("field factor exhaustions complete",flush=True)
    result["local"] = local_exhaustion()
    print("local row and dot-product exhaustions complete",flush=True)
    result["controls"] = adversarial_controls()
    result["scalar_merged"] = scalar_merged_exhaustion()
    print("full scalar merged-instance exhaustion complete",flush=True)
    attacks = [merged_attack(kind) for kind in ("layer","head","both","dimension")]
    result["merged_attacks"] = [{"kind":a["attack"],"wrong_rows":len(a["wrong_rows"]),"every_bracket_passes":a["every_reciprocal_bracket_passes"],"accepted_namespace_variants":[name for name,data in a["variants"].items() if data["full_tuple_identity"]]}for a in attacks]
    for attack in attacks:
        (HERE/"results"/f"counterexample-{attack['attack']}.json").write_text(json.dumps(attack,indent=2)+"\n")
    result["padding"] = padding_checks()
    result["goldilocks"] = goldilocks_obligations()
    # Direct algebraic lifts supplement the fixed finite tests; they are not a new exhaustion.
    lifts = []
    for kind in ("layer","head","both"):
        lifted = merged_attack(kind,positions=4,dimensions=4,modulus=GOLD,base_rows=1<<17)
        lifted["public_parameters"] = {"d_head":4,"n_heads":4,"n_kv_heads":2,"n_layers":2,"sequence_length":4,"score_scale":[1,1],"e_q_plus_e_k":1,"exp_sigma":4,"base_key_bits":17,"doubled_key_bits":18,"spread_bound":128*255*4,"full_model_or_proof_constructed":False}
        assert (128*255*4).bit_length() == 17
        (HERE/"results"/f"counterexample-Goldilocks-{kind}.json").write_text(json.dumps(lifted,indent=2)+"\n")
        lifts.append({"kind":kind,"wrong_rows":len(lifted["wrong_rows"]),"every_bracket_passes":True,"untagged_full_tuple_identity":lifted["variants"]["untagged"]["full_tuple_identity"],"fully_tagged_rejects":not lifted["variants"]["full"]["full_tuple_identity"]})
    result["Goldilocks_direct_lifts"] = lifts
    reciprocal_bounds = {str(context):(2*context-1)*(1 << (2*SIGMA)) for context in (2,4)}
    assert all(bound < F and bound < GOLD for bound in reciprocal_bounds.values())
    result["current_spec_delta"] = {"reviewed_main":"709c37c","change":"#388 adds SoftmaxRow reciprocal uniqueness precondition (2*max_context-1)*2^(2*sigma)<P","sigma":SIGMA,"tested_context_bounds":reciprocal_bounds,"precondition_holds_in_F65537_and_Goldilocks":True,"routing_counterexample_unchanged":True}
    result["verdict"] = "refuted"
    result["reason"] = "Literal merged read identity admits fully balanced wrong/second witnesses; reciprocal brackets included. Scoped local variant has unique witnesses only in the enumerated domains."
    (HERE/"results/run.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"verdict":result["verdict"],"local":result["local"],"scalar_merged_counts":{k:v for k,v in result["scalar_merged"].items() if k!="first_alternative"},"merged_attacks":result["merged_attacks"]},indent=2))


if __name__ == "__main__":
    main()
