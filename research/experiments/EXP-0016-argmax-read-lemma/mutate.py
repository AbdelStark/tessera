#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Isolated relation fault injection; retains scratch work and raw outputs."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    scratch = Path(tempfile.mkdtemp(prefix="tessera-exp16-mutants-"))
    (scratch/"results").mkdir()
    (scratch/"results/README.preregistered.txt").write_bytes((here/"results/README.preregistered.txt").read_bytes())
    original = (here/"lemma.py").read_text().replace("ROOT = HERE.parents[2]", f"ROOT = Path({str(root)!r})")
    cases = [
        ("control",None,None),
        ("routing-tag","return (((l*heads+h)*positions+position)*dimensions+dim)","return position*dimensions+dim"),
        ("parity","- (lt if parity else 0)","- 0"),
        ("padding-zero-pin","return psi != 0 or value % F == 0","return True"),
        ("ceil-to-floor","index = key//2 if floor else (key+1)//2","index = key//2"),
    ]
    outcomes = []
    for name,before,after in cases:
        text = original
        if before is not None:
            assert original.count(before) == 1, name
            text = original.replace(before,after)
        (scratch/"lemma.py").write_text(text)
        run = subprocess.run([sys.executable,"-I",str(scratch/"lemma.py")],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
        (here/"results"/f"mutation-{name}.txt").write_text(run.stdout)
        if name == "control":
            assert run.returncode == 0, run.stdout[-3000:]
            # Preserve the unmodified control's final artifacts and exact environment.
            for filename in ("run.json",):
                (here/"results"/f"control-{filename}").write_bytes((scratch/"results"/filename).read_bytes())
        else:
            assert run.returncode != 0 and "AssertionError" in run.stdout, run.stdout[-3000:]
        outcomes.append({"name":name,"exit_code":run.returncode,"expected":"pass" if name=="control" else "AssertionError"})
    result = {"experiment":"EXP-0016","control_passed":True,"mutants_caught":4,"mutants_total":4,"outcomes":outcomes,"scratch":str(scratch),"original_files_modified":False}
    (here/"results/mutations.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
