---
name: redteam-scheme
description: Offensive cryptanalysis and exploit construction against Tessera's protocol, mathematics, gadgets, commitments, zero knowledge, decoders and repository gates. Use when asked to red-team, attack, break, forge a proof, extract a secret, bypass a gate, or test whether a suspected vulnerability is exploitable.
---

# Red-team the scheme

Think and act as the adversary ([06 §2](../../../docs/spec/06-security.md#2-adversaries)): `ADV-P` (a prover controlling all hardware and software with a large grinding budget who wants the verifier to accept a false statement or a smaller `Û`), `ADV-V` (a malicious verifier choosing seeds, queries and aborts to extract weights, prompts or outputs), `ADV-N` (a network attacker replaying or splicing frames), or `ADV-S` (a supply-chain or build attacker bypassing repository gates). Speculation does not count: the output of this skill is an executable exploit or a proof that the defending check holds.

1. **Choose the target seam and attack class.** Read [attack-catalog.md](attack-catalog.md) and the RFC or crate under attack. Identify the exact check, algebraic identity, range bound, transcript step, resource limit or `xtask` gate you aim to defeat. Done when you can state the target in one sentence: *"As `<ADV>`, by setting `<variables>`, the target at `<file:line or RFC section>` will `<accept false claim / leak secret / OOM / pass gate>`."*
2. **Derive the exploit equation.** Before writing code, solve the constraint system on paper or in Python:
   - For a **soundness attack (`ADV-P`)**: list every equation the verifier actually checks on the path to `Accept`. Find a free variable (an uncommitted witness value, a padding slot, a late-absorbed claim, a canceling batch power, a modular wrap `± k · P`, a zero denominator, a masked tree leaf) and solve for the value that satisfies the verifier's equations while violating the statement.
   - For a **privacy attack (`ADV-V`)**: write the linear or algebraic system relating the revealed transcript values (`Round1`, `Round2Response`, `Round3Response`, error codes or timing) to the secret witness or weights, and find a query choice, Boolean evaluation point (`Z(p) = 0`), or session-abort sequence where the mask rank drops below the number of revealed equations.
   - For a **robustness or gate attack (`ADV-P` / `ADV-S`)**: construct the byte sequence, crate manifest, token stream or dep-info record that reaches the sink before the guard fires.
   Done when the exploit is reduced to an explicit mathematical assignment or byte payload.
3. **Build a standalone exploit in the scratchpad.** Write a self-contained script or test in a scratch copy of the repository (never modifying the working tree directly):
   - Against protocol math or conformance vectors: a Python big-integer script (`python3 -I`) that constructs the forged proof or witness and runs the verifier's check logic.
   - Against `verifier/` (`tv-*`) or `xtask/`: a Rust test or fixture that feeds the malicious payload into the real entry point (`tv_core`, `tv_codec`, `tv_hash`, or `xtask`).
   Done when the script compiles and runs deterministically.
4. **Fire and classify the outcome.**
   - **Kill (exploit succeeds)**: the real verifier returns `Accept` on a false statement, returns the wrong `RejectReason`, panics, allocates past `ResourceLimits`, leaks a secret bit, or `cargo xtask` passes on a policy violation.
   - **Blocked (defense holds)**: the target rejects with the exact expected `RejectReason` or `ErrorCode`. To prove your exploit actually reached the target check (rather than failing on an earlier syntax error), disable only that check in the scratch copy and re-run: the exploit must now succeed.
   Done when you have recorded the exact command and output for both the unmodified target and (if blocked) the disabled-check control.
5. **Immunize the repository.**
   - On a **kill**: minimize the exploit to the smallest input that triggers the bug. Fix the RFC and the implementation together (or, if the fix belongs to another milestone, open a `type:security` issue with the minimized reproducer and block dependent issues, as `#208` and `#237` did). Add the minimized exploit as a permanent regression test in the crate and in `spec/vectors/`.
   - On a **blocked attack** that lacks a dedicated regression test: add the adversarial test to the crate so no future refactor can remove the defense silently.
   - Run `continuous-learning` to add the attack pattern to [attack-catalog.md](attack-catalog.md) and `soundness-review`.
   Done when the regression test is committed (or the blocker issue is filed with its reproducer) and `cargo xtask lint` passes.

## Parallel red-team campaign

To launch six specialized red-team agents in parallel across every attack surface of the repository, run the `redteam-campaign` workflow ([../../workflows/redteam-campaign.js](../../workflows/redteam-campaign.js)).
