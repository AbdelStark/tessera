# EXP-NNNN: <title>

| | |
|---|---|
| Status | `planned` |
| Idea | `IDEA-NNNN`, linked by relative path to `../../ideas/IDEA-NNNN-<slug>.md` |
| Evidence level | <E0 or E1 reachable here; E2 needs the H100 runner> |
| Hardware | <machine; "none" for analytic work> |
| Pre-registered on | YYYY-MM-DD |

## Hypothesis

One falsifiable sentence.

## Success criterion

Written before the code. The number and the comparison that count as `validated`, the one that counts as `refuted`, and what is `inconclusive`. Amendments go below this paragraph with a date; nothing above it is edited after pre-registration.

## Method

- What is implemented, and what the baseline is, in the same harness.
- Inputs: synthetic, from a seeded generator (seed stated).
- Measurement protocol: warm-ups, iterations, the statistic reported (median, minimum, coefficient of variation), as in RFC-0020.
- Correctness check: the digest or value the candidate must reproduce against the reference, so that a fast but wrong implementation cannot pass.

## Reproduction

```sh
# every command, from the repository root
```

Environment: CPU, OS, `rustc --version` or `python3 --version`, date.

## Results

| Case | Baseline | Candidate | Ratio |
|---|---|---|---|
| | | | |

Raw output: `results/` in this directory, or the command above.

## Analysis

The result against the success criterion. Threats to validity: CPU against GPU, operation counts against wall clock, cache effects, input distribution, anything the reader should doubt.

## Verdict

`validated`, `refuted` or `inconclusive`, and the change it makes to the idea's score and status.

## Follow-ups

What the next experiment should test, and what an H100 run would settle.
