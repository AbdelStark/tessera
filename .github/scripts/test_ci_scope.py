# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Keep source, vector, policy and workflow changes on the tested CI path."""

from pathlib import Path
import re
import unittest

from ci_scope import needs_code_checks

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def strip_yaml_comments(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


class ScopeTests(unittest.TestCase):
    def test_serving_integration_guide_runs_its_example(self):
        self.assertTrue(needs_code_checks(["docs/integration.md"]))

    def test_documentation_only(self):
        self.assertFalse(
            needs_code_checks([
                "README.md", "docs/spec/07-testing-strategy.md", "research/tools/cost_model.py"
            ])
        )

    def test_source_mixed_with_docs_and_unknown_paths(self):
        for source in (
            "crates/tessera-field/src/lib.rs", "verifier/crates/tv-core/src/lib.rs",
            "python/tessera_quant/tce.py", "spec/vectors/models/index.json",
            ".github/workflows/ci.yml", ".github/workflows/pr-policy.yml",
            ".github/scripts/ci_scope.py", "Cargo.lock",
            "secret-access.toml", "new-component/input", "odd\nfilename.rs",
            "crates/tessera-compiler/src/schema.md",
        ):
            with self.subTest(source=source):
                self.assertTrue(needs_code_checks(["README.md", source]))

    def test_no_diff_requires_tests(self):
        self.assertTrue(needs_code_checks([]))


class LabelPolicyWorkflowTests(unittest.TestCase):
    """#532: label events run pr-policy alone and never rerun or skip code checks."""

    def test_only_pr_policy_handles_label_events_and_reads_labels(self):
        for path in sorted(WORKFLOWS.glob("*.yml")):
            body = strip_yaml_comments(path.read_text(encoding="utf-8"))
            if path.name == "pr-policy.yml":
                self.assertRegex(
                    body,
                    r"types:\s*\[opened,\s*synchronize,\s*reopened,\s*labeled,\s*unlabeled\]",
                )
                self.assertIn("workflow_dispatch:", body)
                self.assertIn("pull_request.labels", body)
                self.assertNotRegex(body, r"(?m)^concurrency:")
                # A skipped required check counts as passing; pr-policy must never skip.
                jobs = body.split("\njobs:\n", 1)[1]
                self.assertNotRegex(jobs, r"(?m)^\s{4}if:")
            else:
                with self.subTest(workflow=path.name):
                    self.assertNotRegex(body, r"\blabeled\b|\bunlabeled\b")
                    self.assertNotIn("pull_request.labels", body)

    def test_ci_runs_on_commit_lifecycle_events_only(self):
        body = strip_yaml_comments((WORKFLOWS / "ci.yml").read_text(encoding="utf-8"))
        self.assertRegex(body, r"types:\s*\[opened,\s*synchronize,\s*reopened\]")
        self.assertNotIn("check-pr", body)

