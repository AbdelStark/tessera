# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Keep source, vector, policy and workflow changes on the tested CI path."""

import unittest

from ci_scope import needs_code_checks


class ScopeTests(unittest.TestCase):
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
            ".github/workflows/ci.yml", ".github/scripts/ci_scope.py", "Cargo.lock",
            "secret-access.toml", "new-component/input", "odd\nfilename.rs",
            "crates/tessera-compiler/src/schema.md",
        ):
            with self.subTest(source=source):
                self.assertTrue(needs_code_checks(["README.md", source]))

    def test_no_diff_requires_tests(self):
        self.assertTrue(needs_code_checks([]))
