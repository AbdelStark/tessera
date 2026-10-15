# Security policy

Tessera's verifier certifies statements between parties that do not trust each other. A flaw that lets an invalid proof pass, or that lets a proof reveal what it should hide, defeats the purpose of the system, so security reports are handled before any other work.

## Reporting a vulnerability

Report vulnerabilities privately, through GitHub's private vulnerability reporting: open this repository's **Security** tab and choose **Report a vulnerability**. While the repository is private, that option is not available; collaborators open a draft security advisory under **Security → Advisories** instead. Keep vulnerabilities out of public issues, pull requests and discussions until a fix is released.

A useful report names the affected component and version or commit, describes the impact, and includes a reproduction: for example, a proof or artifact that the verifier accepts but should reject, or an input that crashes it.

## What to expect

| Step | Target |
|---|---|
| Acknowledgement | within 3 business days |
| Triage: severity and plan | within 10 business days |
| Credit | in the advisory and the release notes, unless you ask otherwise |

## Severity

| Severity | Examples |
|---|---|
| Critical | Any soundness break: the verifier accepts an invalid proof or a false statement. Any zero-knowledge break: a proof reveals weights, prompts, outputs or activations beyond the public statement. |
| High | Resource exhaustion of the verifier on hostile input: unbounded memory or time, or a crash. |
| Medium | A prover-side crash on valid input. |

Other issues are triaged case by case.

## Fixes and disclosure

- Critical issues ship as a patch release. When the fix changes verification, it also ships a new protocol version ([09 §3](docs/spec/09-release-and-versioning.md#3-protocol-and-schema-versions)), and the release notes state which earlier proofs are affected.
- Security fixes are listed in the changelog under "Security", with their advisory identifier.

## Scope

Everything in this repository is in scope: the prover, the independent verifier, the quantization front-end, the specification and the build and release tooling. As stated in the threat model ([06 §2](docs/spec/06-security.md#2-adversaries)), an adversary who subverts the recorder itself and physical attacks on the verifier host are out of scope.

This policy implements [06 §8](docs/spec/06-security.md#8-disclosure-policy).
